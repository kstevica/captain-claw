// The ghost's body (FPV plan Phase 1): a three.js first-person walk
// through the block village. Borrowed from the VOXELHEIM engine skeleton —
// pointer lock, WASD with sub-stepped AABB collision, one-pass meshing,
// procedural sky — with everything survival stripped. The sun follows the
// REAL clock: visit at dusk and the village is at dusk, lamps lit.

import * as THREE from 'three'
import type { GhostPresence, VillageBeingPos, VillageNote, VillagePlace } from '../../../services/beings'
import { Figures } from './figures'
import { Ghosts } from './ghosts'
import { Lecterns } from './lecterns'
import { Signs } from './signs'
import { ATLAS_STEP, B, BLOCK_TILES, buildAtlas, isGlow, isSolid } from './textures'
import type { BuiltWorld, Lectern } from './worldgen'
import { D, H, MARGIN, SURFACE, W } from './worldgen'

export interface FPVStatus {
  place: string          // "at the Library" / "on the street" / …
  phase: boolean         // phase mode (fly + noclip)
  note: VillageNote | null   // the sign underfoot, if any (Phase 3)
  readable: Lectern | null   // the reading stand within reach (Phase 4)
}

export interface FPVHooks {
  onLock: (locked: boolean) => void
  onStatus: (s: FPVStatus) => void
  // Phase 3: E plants a sign at the ghost's feet; X pulls the one underfoot
  onPlant?: (units: { x: number; y: number }) => void
  onPull?: (note: VillageNote) => void
  // Phase 4: R opens the reader for the stand within reach
  onRead?: (lectern: Lectern) => void
}

export interface FPVHandle {
  lock: () => void
  dispose: () => void
  // a fresh map payload: the world stays, the walkers update (Phase 2)
  setBeings: (beings: VillageBeingPos[], places: VillagePlace[], fetchedAtMs: number) => void
  setNotes: (notes: VillageNote[]) => void
  // the other ghosts here right now (Phase 5): parent + visitors
  setGhosts: (ghosts: GhostPresence[]) => void
  // where the ghost stands, in village units (for presence + planting)
  positionUnits: () => { x: number; y: number }
  // touch / mobile (Phase 6): roam without a mouse
  enterTouch: () => void                          // step in, no pointer lock
  setMove: (x: number, y: number) => void         // joystick (-1..1 each)
  look: (dyaw: number, dpitch: number) => void    // drag-to-look (radians)
  jump: () => void
  toggleFly: () => boolean                         // returns the new phase state
  note: () => void
  read: () => boolean                              // true if a stand was in reach
  setGyro: (on: boolean) => void                   // steer look by phone tilt
}

const clamp = (v: number, a: number, b: number) => (v < a ? a : v > b ? b : v)
const lerp = (a: number, b: number, t: number) => a + (b - a) * t
const smooth = (a: number, b: number, v: number) => {
  const t = clamp((v - a) / (b - a), 0, 1); return t * t * (3 - 2 * t)
}

// face definitions: direction, corners, storybook light per face
const FACES = [
  { d: [1, 0, 0], c: [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]], sh: 0.74 },
  { d: [-1, 0, 0], c: [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]], sh: 0.74 },
  { d: [0, 1, 0], c: [[0, 1, 1], [1, 1, 1], [1, 1, 0], [0, 1, 0]], sh: 1.0 },
  { d: [0, -1, 0], c: [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], sh: 0.55 },
  { d: [0, 0, 1], c: [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], sh: 0.87 },
  { d: [0, 0, -1], c: [[1, 0, 0], [0, 0, 0], [0, 1, 0], [1, 1, 0]], sh: 0.87 },
] as const
const FUV = [[0, 0], [1, 0], [1, 1], [0, 1]] as const

// one mesh for the lit world, one for lamp-glass that ignores the night
function buildMeshes(world: BuiltWorld, atlas: THREE.CanvasTexture,
                     uvFor: (t: number) => [number, number]) {
  const P: number[] = [], N: number[] = [], U: number[] = [], C: number[] = [], I: number[] = []
  const gP: number[] = [], gN: number[] = [], gU: number[] = [], gI: number[] = []
  for (let y = 0; y < H; y++) for (let z = 0; z < D; z++) for (let x = 0; x < W; x++) {
    const id = world.get(x, y, z)
    if (id === B.AIR) continue
    const glow = isGlow(id)
    const tiles = BLOCK_TILES[id]
    for (let f = 0; f < FACES.length; f++) {
      const F = FACES[f]
      const nb = world.get(x + F.d[0], y + F.d[1], z + F.d[2])
      // water renders only its exposed top; solids cull against solids
      if (id === B.WATER) { if (f !== 2 || nb !== B.AIR) continue }
      else if (nb !== B.AIR && nb !== B.WATER) continue
      const t = tiles[f === 2 ? 0 : f === 3 ? 2 : 1]
      const [u0, v0] = uvFor(t)
      const e = 0.0015
      if (glow) {
        const b0 = gP.length / 3
        for (let i = 0; i < 4; i++) {
          const cc = F.c[i]
          gP.push(x + cc[0], y + cc[1], z + cc[2])
          gN.push(F.d[0], F.d[1], F.d[2])
          gU.push(u0 + e + FUV[i][0] * (ATLAS_STEP - 2 * e), v0 + e + FUV[i][1] * (ATLAS_STEP - 2 * e))
        }
        gI.push(b0, b0 + 1, b0 + 2, b0, b0 + 2, b0 + 3)
      } else {
        const b0 = P.length / 3
        // a whisper of per-block variance keeps big lawns from banding
        const jit = id === B.GRASS || id === B.MEADOW
          ? 0.94 + ((x * 7 + z * 13) % 5) * 0.03 : 1
        for (let i = 0; i < 4; i++) {
          const cc = F.c[i]
          P.push(x + cc[0], y + cc[1], z + cc[2])
          N.push(F.d[0], F.d[1], F.d[2])
          U.push(u0 + e + FUV[i][0] * (ATLAS_STEP - 2 * e), v0 + e + FUV[i][1] * (ATLAS_STEP - 2 * e))
          C.push(F.sh * jit, F.sh * jit, F.sh * jit)
        }
        I.push(b0, b0 + 1, b0 + 2, b0, b0 + 2, b0 + 3)
      }
    }
  }
  const solidGeo = new THREE.BufferGeometry()
  solidGeo.setAttribute('position', new THREE.Float32BufferAttribute(P, 3))
  solidGeo.setAttribute('normal', new THREE.Float32BufferAttribute(N, 3))
  solidGeo.setAttribute('uv', new THREE.Float32BufferAttribute(U, 2))
  solidGeo.setAttribute('color', new THREE.Float32BufferAttribute(C, 3))
  solidGeo.setIndex(I)
  const solid = new THREE.Mesh(solidGeo,
    new THREE.MeshLambertMaterial({ map: atlas, vertexColors: true }))
  const glowGeo = new THREE.BufferGeometry()
  glowGeo.setAttribute('position', new THREE.Float32BufferAttribute(gP, 3))
  glowGeo.setAttribute('normal', new THREE.Float32BufferAttribute(gN, 3))
  glowGeo.setAttribute('uv', new THREE.Float32BufferAttribute(gU, 2))
  glowGeo.setIndex(gI)
  const glow = new THREE.Mesh(glowGeo, new THREE.MeshBasicMaterial({ map: atlas }))
  return { solid, glow }
}

export function createFPV(canvas: HTMLCanvasElement, world: BuiltWorld,
                          hooks: FPVHooks): FPVHandle {
  // ── scene ────────────────────────────────────────────────────────────
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: false })
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  const scene = new THREE.Scene()
  const camera = new THREE.PerspectiveCamera(75, 1, 0.08, 600)
  camera.rotation.order = 'YXZ'
  scene.fog = new THREE.Fog(0x9cc4e8, 60, 220)

  const { texture: atlas, uvFor } = buildAtlas()
  const { solid, glow } = buildMeshes(world, atlas, uvFor)
  scene.add(solid); scene.add(glow)

  const hemi = new THREE.HemisphereLight(0xdfeaff, 0x9a8f78, 0.9)
  const sun = new THREE.DirectionalLight(0xffffff, 1.5)
  scene.add(hemi, sun, sun.target)

  // the Iskre themselves — paper cutouts on the true clock (Phase 2)
  const figures = new Figures(scene)
  // planted signs in the grass (Phase 3)
  const signs = new Signs(scene)
  // reading stands inside the buildings (Phase 4)
  const lecterns = new Lecterns(scene, world.lecterns)
  // the other ghosts roaming with you (Phase 5)
  const ghosts = new Ghosts(scene)
  const toUnits = () => ({
    x: Math.min(1000, Math.max(0, Math.round((player.pos.x - MARGIN) * 5))),
    y: Math.min(1000, Math.max(0, Math.round((player.pos.z - MARGIN) * 5))),
  })

  // sky pivot: sun disc, moon disc, stars — turned by the REAL clock
  const skyPivot = new THREE.Group()
  scene.add(skyPivot)
  const sunMesh = new THREE.Mesh(new THREE.PlaneGeometry(26, 26),
    new THREE.MeshBasicMaterial({ color: 0xffe27a, fog: false }))
  sunMesh.position.set(300, 0, 80); skyPivot.add(sunMesh)
  const moonMesh = new THREE.Mesh(new THREE.PlaneGeometry(18, 18),
    new THREE.MeshBasicMaterial({ color: 0xe8eeff, fog: false }))
  moonMesh.position.set(-300, 0, -80); skyPivot.add(moonMesh)
  const starGeo = new THREE.BufferGeometry()
  {
    const pos: number[] = []
    let s = 0x517A
    const rng = () => { s = (s * 16807) % 2147483647; return s / 2147483647 }
    for (let i = 0; i < 360; i++) {
      const a = rng() * Math.PI * 2, e2 = Math.acos(rng() * 2 - 1)
      pos.push(340 * Math.sin(e2) * Math.cos(a), 340 * Math.cos(e2), 340 * Math.sin(e2) * Math.sin(a))
    }
    starGeo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
  }
  const starMat = new THREE.PointsMaterial({ color: 0xcdd8ff, size: 1.6, sizeAttenuation: false, transparent: true, opacity: 0, fog: false })
  const stars = new THREE.Points(starGeo, starMat)
  skyPivot.add(stars)

  const cloudMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.5, fog: false })
  const cloudGeo = new THREE.BoxGeometry(1, 1, 1)
  const clouds: THREE.Mesh[] = []
  for (let i = 0; i < 14; i++) {
    const m = new THREE.Mesh(cloudGeo, cloudMat)
    const h2 = (n: number) => ((i * 2654435761 + n * 40503) % 1000) / 1000
    m.scale.set(14 + h2(1) * 22, 1.6, 9 + h2(2) * 12)
    m.position.set(h2(3) * W, 46 + h2(4) * 10, h2(5) * D)
    scene.add(m); clouds.push(m)
  }

  // ── the ghost ────────────────────────────────────────────────────────
  const player = {
    pos: { x: world.spawn.x, y: world.spawn.y, z: world.spawn.z },
    vel: { x: 0, y: 0, z: 0 },
    w: 0.3, h: 1.8, onGround: false,
  }
  let yaw = world.spawn.yaw, pitch = -0.05
  let phase = false                 // phase mode: fly + drift through walls
  let bobPhase = 0
  const keys: Record<string, boolean> = {}
  let locked = false
  // when the document refuses pointer lock (embedded panes, iframes) the
  // ghost still walks: soft mode — drag to look, Esc handled by hand
  let soft = false
  let dragging = false
  // touch / mobile (Phase 6): the joystick vector, a queued jump, and the
  // phone's own orientation when the gyro look is on
  const touch = { x: 0, y: 0 }      // joystick: x = strafe, y = forward
  let jumpQueued = false
  let gyro = false

  const solidAt = (x: number, y: number, z: number) => isSolid(world.get(x, y, z))

  function moveAxis(axis: 'x' | 'y' | 'z', amt: number): boolean {
    if (!amt) return false
    const p = player.pos
    p[axis] += amt
    const w2 = player.w, h2 = player.h, EPS = 1e-3
    const x0 = Math.floor(p.x - w2), x1 = Math.floor(p.x + w2 - 1e-9)
    const y0 = Math.floor(p.y), y1 = Math.floor(p.y + h2 - 1e-9)
    const z0 = Math.floor(p.z - w2), z1 = Math.floor(p.z + w2 - 1e-9)
    for (let y = y0; y <= y1; y++) for (let z = z0; z <= z1; z++) for (let x = x0; x <= x1; x++) {
      if (!solidAt(x, y, z)) continue
      if (axis === 'x') p.x = amt > 0 ? x - w2 - EPS : x + 1 + w2 + EPS
      else if (axis === 'y') p.y = amt > 0 ? y - h2 - EPS : y + 1 + EPS
      else p.z = amt > 0 ? z - w2 - EPS : z + 1 + w2 + EPS
      return true
    }
    return false
  }
  function entityStep(dt: number) {
    const maxV = Math.max(Math.abs(player.vel.x), Math.abs(player.vel.y), Math.abs(player.vel.z))
    const sub = Math.max(1, Math.ceil((maxV * dt) / 0.4)), sdt = dt / sub
    player.onGround = false
    for (let i = 0; i < sub; i++) {
      if (moveAxis('y', player.vel.y * sdt)) { if (player.vel.y < 0) player.onGround = true; player.vel.y = 0 }
      if (moveAxis('x', player.vel.x * sdt)) player.vel.x = 0
      if (moveAxis('z', player.vel.z * sdt)) player.vel.z = 0
    }
  }

  function updatePlayer(dt: number) {
    // keyboard + the touch joystick blend into one move vector
    const f = clamp((keys.KeyW ? 1 : 0) - (keys.KeyS ? 1 : 0) + touch.y, -1, 1)
    const s = clamp((keys.KeyD ? 1 : 0) - (keys.KeyA ? 1 : 0) + touch.x, -1, 1)
    let fx = -Math.sin(yaw) * f + Math.cos(yaw) * s
    let fz = -Math.cos(yaw) * f - Math.sin(yaw) * s
    const fl = Math.hypot(fx, fz) || 1
    fx /= fl; fz /= fl
    const moving = Math.hypot(f, s) > 0.12

    if (phase) {
      // the ghost remembers it is a ghost: drift, rise, sink — no walls.
      // a queued touch-jump nudges up; Shift/joystick pull is the down.
      const speed = 11
      const up = (keys.Space || jumpQueued ? 1 : 0) - (keys.ShiftLeft || keys.ShiftRight ? 1 : 0)
      jumpQueued = false
      player.pos.x = clamp(player.pos.x + (moving ? fx * speed * dt : 0), 1, W - 1)
      player.pos.z = clamp(player.pos.z + (moving ? fz * speed * dt : 0), 1, D - 1)
      player.pos.y = clamp(player.pos.y + up * speed * 0.8 * dt, 1, H + 14)
      player.vel.x = player.vel.y = player.vel.z = 0
    } else {
      const sprint = keys.ShiftLeft || keys.ShiftRight
      const speed = sprint ? 6.4 : 4.4
      const accel = player.onGround ? 12 : 4
      const k = Math.min(1, accel * dt)
      player.vel.x += ((moving ? fx * speed : 0) - player.vel.x) * k
      player.vel.z += ((moving ? fz * speed : 0) - player.vel.z) * k
      player.vel.y -= 30 * dt
      if ((keys.Space || jumpQueued) && player.onGround) player.vel.y = 9.2
      jumpQueued = false
      player.vel.y = Math.max(player.vel.y, -38)
      entityStep(dt)
      // fell out of the world somehow → walk home to spawn
      if (player.pos.y < -10) { player.pos = { ...world.spawn, y: SURFACE + 2 } as typeof player.pos; player.vel.y = 0 }
    }

    const hspeed = Math.hypot(player.vel.x, player.vel.z)
    if (player.onGround && hspeed > 1) bobPhase += dt * hspeed * 1.7
    const bob = !phase && player.onGround && hspeed > 1 ? Math.sin(bobPhase * 2) * 0.05 : 0
    camera.position.set(player.pos.x, player.pos.y + 1.62 + bob, player.pos.z)
    camera.rotation.y = yaw
    camera.rotation.x = pitch
  }

  // ── the sky follows the real clock ───────────────────────────────────
  const colDay = new THREE.Color(0x9cc4e8), colNight = new THREE.Color(0x10152b)
  const colDusk = new THREE.Color(0xffab6e)
  const tmp = new THREE.Color(), tmp2 = new THREE.Color()
  function updateSky() {
    const d0 = new Date()
    const hr = d0.getHours() + d0.getMinutes() / 60 + d0.getSeconds() / 3600
    const tod = (((hr - 6) / 24) % 1 + 1) % 1     // 0 = sunrise, .25 = noon
    const a = tod * Math.PI * 2
    const el = Math.sin(a)
    const dayF = smooth(-0.12, 0.14, el)
    const horizon = clamp(1 - Math.abs(el) * 3.2, 0, 1) * 0.6
    tmp.copy(colNight).lerp(colDay, dayF)
    tmp2.copy(tmp).lerp(colDusk, horizon * (0.25 + dayF * 0.75))
    scene.background = tmp2
    ;(scene.fog as THREE.Fog).color.copy(tmp2)
    hemi.intensity = 0.55 + dayF * 1.25
    sun.intensity = 0.18 + dayF * 2.3
    sun.color.setHSL(0.12, 0.5, lerp(0.6, 0.95, 1 - horizon))
    skyPivot.rotation.z = a
    skyPivot.position.set(player.pos.x, player.pos.y, player.pos.z)
    sunMesh.lookAt(camera.position)
    moonMesh.lookAt(camera.position)
    starMat.opacity = (1 - dayF) * 0.9
    cloudMat.color.setScalar(lerp(0.3, 1, dayF))
    const sd = Math.cos(a), se = Math.sin(a)
    const useMoon = se < 0
    sun.position.set(player.pos.x + (useMoon ? -sd : sd) * 120,
      player.pos.y + Math.abs(se) * 120 + 16, player.pos.z + 40)
    sun.target.position.set(player.pos.x, player.pos.y, player.pos.z)
  }

  // ── where am I? (the location chip) ──────────────────────────────────
  let lastStatus = ''
  function updateStatus() {
    const px = player.pos.x, pz = player.pos.z
    let place = ''
    let nearest = ''
    let nearestD = Infinity
    for (const l of world.labels) {
      if (px >= l.x0 - 1 && px <= l.x1 + 2 && pz >= l.z0 - 1 && pz <= l.z1 + 2) { place = `at ${l.name}`; break }
      const dx = Math.max(l.x0 - px, 0, px - l.x1 - 1)
      const dz = Math.max(l.z0 - pz, 0, pz - l.z1 - 1)
      const d2 = Math.hypot(dx, dz)
      if (d2 < nearestD) { nearestD = d2; nearest = l.name }
    }
    if (!place) {
      const under = world.get(Math.floor(px), Math.floor(player.pos.y) - 1, Math.floor(pz))
      if (under === B.PATH) place = 'on the street'
      else if (nearestD < 10) place = `near ${nearest}`
      else if (px < 8 || px > W - 8 || pz < 8 || pz > D - 8) place = 'at the edge of the woods'
      else place = 'on the open green'
    }
    const note = signs.nearest({ x: px, z: pz })
    const readable = lecterns.nearest({ x: px, z: pz })
    const key = `${place}|${phase}|${note?.id ?? ''}|${readable?.placeId ?? ''}`
    if (key !== lastStatus) {
      lastStatus = key
      hooks.onStatus({ place, phase, note, readable })
    }
  }

  // ── input ────────────────────────────────────────────────────────────
  // when the ghost is typing (a note, a name) the game keeps its hands off
  // the keyboard — otherwise Space (jump) eats the space, and every letter
  // is a command instead of a character.
  const typing = () => {
    const el = document.activeElement as HTMLElement | null
    return !!el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA'
      || el.tagName === 'SELECT' || el.isContentEditable)
  }
  // the actions the keys (and the touch buttons) fire — one place each, so
  // a tap and a keypress are exactly the same thing.
  const releasePointer = () => {
    if (document.pointerLockElement === canvas) document.exitPointerLock()
    else if (soft) { locked = false; soft = false; dragging = false; onBlur(); hooks.onLock(false) }
  }
  const doFly = () => { phase = !phase; player.vel.y = 0; lastStatus = ''; return phase }
  const doNote = () => { if (!hooks.onPlant) return; releasePointer(); hooks.onPlant(toUnits()) }
  const doRead = () => {
    if (!hooks.onRead) return false
    const stand = lecterns.nearest({ x: player.pos.x, z: player.pos.z })
    if (!stand) return false
    releasePointer(); hooks.onRead(stand); return true
  }
  const doPull = () => {
    if (!hooks.onPull) return
    const near = signs.nearest({ x: player.pos.x, z: player.pos.z })
    if (near) hooks.onPull(near)
  }
  const onKeyDown = (e: KeyboardEvent) => {
    if (typing()) return
    if (e.code === 'Space') e.preventDefault()
    keys[e.code] = true
    if (!locked) {
      if (e.code === 'Escape' && soft) { locked = false; soft = false; dragging = false; onBlur(); hooks.onLock(false) }
      return
    }
    if (e.code === 'KeyF') doFly()
    else if (e.code === 'KeyE') doNote()
    else if (e.code === 'KeyX') doPull()
    else if (e.code === 'KeyR') doRead()
    else if (e.code === 'Escape' && soft) {
      // no real lock to release — pause by hand
      locked = false; soft = false; dragging = false; onBlur()
      hooks.onLock(false)
    }
  }
  const onKeyUp = (e: KeyboardEvent) => { if (!typing()) keys[e.code] = false }
  const onBlur = () => { for (const k in keys) keys[k] = false; touch.x = touch.y = 0 }

  // gyro look (Phase 6): the phone's own orientation steers the view. A
  // baseline is captured when it turns on, so enabling doesn't snap the
  // camera — we apply the delta from where the phone was held. Works in
  // portrait and (baselined) landscape: alpha → yaw, beta → pitch.
  let gyroBase: { yaw: number; pitch: number; a: number; b: number } | null = null
  const onOrient = (e: DeviceOrientationEvent) => {
    if (!gyro || e.alpha == null || e.beta == null) return
    const a = (e.alpha * Math.PI) / 180, b = (e.beta * Math.PI) / 180
    if (!gyroBase) { gyroBase = { yaw, pitch, a, b }; return }
    let da = a - gyroBase.a
    while (da > Math.PI) da -= Math.PI * 2
    while (da < -Math.PI) da += Math.PI * 2
    yaw = gyroBase.yaw + da
    pitch = clamp(gyroBase.pitch - (b - gyroBase.b), -1.45, 1.45)
  }
  const onMouseMove = (e: MouseEvent) => {
    if (!locked || (soft && !dragging)) return
    yaw -= e.movementX * 0.0022
    pitch = clamp(pitch - e.movementY * 0.0022, -1.55, 1.55)
  }
  const onMouseDown = (e: MouseEvent) => { if (soft && locked && e.target === canvas) dragging = true }
  const onMouseUp = () => { dragging = false }
  const onLockChange = () => {
    if (soft) return
    locked = document.pointerLockElement === canvas
    if (!locked) onBlur()
    hooks.onLock(locked)
  }
  document.addEventListener('keydown', onKeyDown)
  document.addEventListener('keyup', onKeyUp)
  document.addEventListener('mousemove', onMouseMove)
  document.addEventListener('mousedown', onMouseDown)
  document.addEventListener('mouseup', onMouseUp)
  document.addEventListener('pointerlockchange', onLockChange)
  window.addEventListener('blur', onBlur)

  // ── size to the canvas box ───────────────────────────────────────────
  const resize = () => {
    const w2 = canvas.clientWidth || 1, h2 = canvas.clientHeight || 1
    renderer.setSize(w2, h2, false)
    camera.aspect = w2 / h2
    camera.updateProjectionMatrix()
  }
  resize()
  const ro = new ResizeObserver(resize)
  ro.observe(canvas)

  // ── the loop ─────────────────────────────────────────────────────────
  let last = performance.now()
  let statusT = 0
  const frame = (now: number) => {
    const dt = clamp((now - last) / 1000, 0, 0.05)
    last = now
    if (locked) updatePlayer(dt)
    else camera.position.set(player.pos.x, player.pos.y + 1.62, player.pos.z)
    updateSky()
    figures.update(dt, player.pos)
    signs.update(player.pos)
    lecterns.update(now / 1000, player.pos)
    ghosts.update(dt, player.pos)
    statusT -= dt
    if (statusT <= 0) { statusT = 0.25; updateStatus() }
    for (const c of clouds) {
      c.position.x += dt * 1.2
      if (c.position.x > W + 40) c.position.x = -40
    }
    renderer.render(scene, camera)
  }
  renderer.setAnimationLoop(frame)

  const softEnter = () => {
    if (locked) return
    soft = true; locked = true
    hooks.onLock(true)
  }
  // a quiet debug handle for live verification (no gameplay use):
  // step() drives frames by hand where RAF is throttled (hidden panes)
  ;(window as unknown as Record<string, unknown>).__fpv = {
    player, get locked() { return locked }, get soft() { return soft },
    get keys() { return keys }, get yaw() { return yaw },
    look: (dy: number, dp: number) => { yaw += dy; pitch = clamp(pitch + dp, -1.55, 1.55) },
    step: (ms = 100, n = 1) => { for (let i = 0; i < n; i++) frame(last + ms) },
    setMove: (x: number, y: number) => { touch.x = clamp(x, -1, 1); touch.y = clamp(y, -1, 1) },
    jump: () => { jumpQueued = true },
    get phase() { return phase },
  }
  return {
    lock: () => {
      if (locked) return
      try {
        const p = canvas.requestPointerLock() as unknown as Promise<void> | undefined
        if (p && typeof p.catch === 'function') p.catch(softEnter)
        else if (!p) window.setTimeout(() => { if (document.pointerLockElement !== canvas) softEnter() }, 250)
      } catch { softEnter() }
    },
    setBeings: (beings, places, fetchedAtMs) => figures.sync(beings, places, fetchedAtMs),
    setNotes: (notes) => { signs.sync(notes); lastStatus = '' },
    setGhosts: (roster) => ghosts.sync(roster),
    positionUnits: toUnits,
    enterTouch: () => { if (!locked) { soft = true; locked = true; hooks.onLock(true) } },
    setMove: (x, y) => { touch.x = clamp(x, -1, 1); touch.y = clamp(y, -1, 1) },
    look: (dyaw, dpitch) => { yaw -= dyaw; pitch = clamp(pitch - dpitch, -1.55, 1.55) },
    jump: () => { jumpQueued = true },
    toggleFly: () => doFly(),
    note: () => doNote(),
    read: () => doRead(),
    setGyro: (on) => {
      gyro = on
      gyroBase = null
      if (on) window.addEventListener('deviceorientation', onOrient)
      else window.removeEventListener('deviceorientation', onOrient)
    },
    dispose: () => {
      renderer.setAnimationLoop(null)
      document.removeEventListener('keydown', onKeyDown)
      document.removeEventListener('keyup', onKeyUp)
      document.removeEventListener('mousemove', onMouseMove)
      document.removeEventListener('mousedown', onMouseDown)
      document.removeEventListener('mouseup', onMouseUp)
      document.removeEventListener('pointerlockchange', onLockChange)
      window.removeEventListener('deviceorientation', onOrient)
      window.removeEventListener('blur', onBlur)
      ro.disconnect()
      if (document.pointerLockElement === canvas) document.exitPointerLock()
      figures.dispose()
      signs.dispose()
      lecterns.dispose()
      ghosts.dispose()
      solid.geometry.dispose(); (solid.material as THREE.Material).dispose()
      glow.geometry.dispose(); (glow.material as THREE.Material).dispose()
      starGeo.dispose(); starMat.dispose()
      cloudGeo.dispose(); cloudMat.dispose()
      sunMesh.geometry.dispose(); (sunMesh.material as THREE.Material).dispose()
      moonMesh.geometry.dispose(); (moonMesh.material as THREE.Material).dispose()
      atlas.dispose()
      renderer.dispose()
    },
  }
}
