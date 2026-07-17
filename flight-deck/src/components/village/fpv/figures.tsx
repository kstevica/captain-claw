// The Iskre, on paper (FPV plan Phase 2): each being's SVG avatar is
// rasterized once — with a cream sticker outline, so it reads as a paper
// cutout standing in the voxel world — onto a flat plane that billboards
// toward the ghost (y-axis only, storybook style: identity always
// readable). They walk the SAME polylines the 2D map animates (walk.posOf,
// zero polling), at their true living pace. Come within ~3 tiles and they
// stop mid-step, give a little hop, and a spark blinks over their head:
// they can't see you. Not exactly.

import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import * as THREE from 'three'
import type { VillageBeingPos, VillagePlace } from '../../../services/beings'
import { IskraAvatar, PALETTES } from '../avatars'
import { posOf, type PlaceById } from '../walk'
import { MARGIN, SURFACE } from './worldgen'

const UNITS_PER_BLOCK = 5
const SENSE_RADIUS = 12          // blocks (~3 tiles): near enough to feel
const SENSE_HOLD_S = 3.5         // how long they stand still, listening
const SENSE_COOLDOWN_S = 60      // per being — a ghost is not an alarm
const BLEND_S = 1.5              // easing back onto the true clock

// stage → paper size (height in blocks; width follows the 48:64 viewBox)
const STAGE_SCALE: Record<string, number> = {
  infant: 0.6, child: 0.8, adolescent: 0.92,
}
const FIG_H = 1.7

// ── rasterization: SVG avatar → sticker-outlined canvas texture ─────────
const texCache = new Map<string, Promise<THREE.CanvasTexture>>()

function rasterize(c: number, p: string): Promise<THREE.CanvasTexture> {
  const key = `${c}:${p}`
  const got = texCache.get(key)
  if (got) return got
  const made = new Promise<THREE.CanvasTexture>((resolve, reject) => {
    // React omits xmlns; a standalone <img>-loaded SVG document requires it
    const svg = renderToStaticMarkup(createElement(IskraAvatar, { c, p, size: 48 }))
      .replace('<svg ', '<svg xmlns="http://www.w3.org/2000/svg" ')
    const img = new Image()
    img.onload = () => {
      const W = 168, H = 224, PAD = 10   // 48×64 ×3.5, room for the outline
      const cv = document.createElement('canvas')
      cv.width = W; cv.height = H
      const ctx = cv.getContext('2d')!
      // the sticker edge: stamp the silhouette 8 ways, tint it cream
      const inner = document.createElement('canvas')
      inner.width = W; inner.height = H
      const ictx = inner.getContext('2d')!
      ictx.drawImage(img, PAD, PAD, W - PAD * 2, H - PAD * 2)
      ictx.globalCompositeOperation = 'source-in'
      ictx.fillStyle = '#f2ead2'
      ictx.fillRect(0, 0, W, H)
      const R = 4
      for (let a = 0; a < 8; a++) {
        ctx.drawImage(inner, Math.cos(a * Math.PI / 4) * R, Math.sin(a * Math.PI / 4) * R)
      }
      ctx.drawImage(img, PAD, PAD, W - PAD * 2, H - PAD * 2)
      const tex = new THREE.CanvasTexture(cv)
      tex.colorSpace = THREE.SRGBColorSpace
      tex.magFilter = THREE.LinearFilter
      tex.minFilter = THREE.LinearFilter
      tex.generateMipmaps = false
      resolve(tex)
    }
    img.onerror = () => reject(new Error('avatar rasterize failed'))
    img.src = 'data:image/svg+xml;utf8,' + encodeURIComponent(svg)
  })
  texCache.set(key, made)
  return made
}

function nameTag(text: string): THREE.Sprite {
  const cv = document.createElement('canvas')
  cv.width = 256; cv.height = 64
  const ctx = cv.getContext('2d')!
  ctx.font = '600 26px -apple-system, system-ui, sans-serif'
  const w = Math.min(236, ctx.measureText(text).width + 28)
  const x0 = (256 - w) / 2
  ctx.fillStyle = 'rgba(23,20,16,0.82)'
  ctx.beginPath()
  ctx.roundRect(x0, 12, w, 40, 20)
  ctx.fill()
  ctx.fillStyle = '#e8e2cf'
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle'
  ctx.fillText(text, 128, 33)
  const tex = new THREE.CanvasTexture(cv)
  tex.colorSpace = THREE.SRGBColorSpace
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false }))
  sp.scale.set(1.7, 0.42, 1)
  return sp
}

function spark(): THREE.Sprite {
  const cv = document.createElement('canvas')
  cv.width = cv.height = 64
  const ctx = cv.getContext('2d')!
  ctx.font = '44px -apple-system, system-ui, sans-serif'
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle'
  ctx.fillText('✦', 32, 34)
  ctx.fillStyle = '#ffdfae'
  ctx.globalCompositeOperation = 'source-in'
  ctx.fillRect(0, 0, 64, 64)
  const tex = new THREE.CanvasTexture(cv)
  tex.colorSpace = THREE.SRGBColorSpace
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false, opacity: 0 }))
  sp.scale.set(0.4, 0.4, 1)
  return sp
}

interface Figure {
  group: THREE.Group
  paper: THREE.Mesh
  mat: THREE.MeshBasicMaterial
  tag: THREE.Sprite
  mark: THREE.Sprite
  b: VillageBeingPos
  avatarKey: string
  h: number
  bobPhase: number
  lastXZ: [number, number] | null
  sensedAt: number       // seconds-clock of the last sense trigger, -∞ never
  frozen: [number, number] | null
}

export class Figures {
  private scene: THREE.Scene
  private bySlug = new Map<string, Figure>()
  private placeById: PlaceById = {}
  private fetchedAtMs = 0
  private clock = 0      // engine seconds — steps even where RAF throttles

  constructor(scene: THREE.Scene) {
    this.scene = scene
  }

  // a fresh map payload: add the new, retire the gone, restyle the changed
  sync(beings: VillageBeingPos[], places: VillagePlace[], fetchedAtMs: number) {
    this.placeById = Object.fromEntries(places.map((p) => [p.id, p]))
    this.fetchedAtMs = fetchedAtMs
    const seen = new Set<string>()
    for (const b of beings) {
      seen.add(b.slug)
      const have = this.bySlug.get(b.slug)
      if (have) {
        have.b = b
        const key = `${b.avatar?.c ?? 1}:${b.avatar?.p ?? 'ember'}`
        if (key !== have.avatarKey) {
          have.avatarKey = key
          void rasterize(b.avatar?.c ?? 1, b.avatar?.p ?? 'ember')
            .then((tex) => { have.mat.map = tex; have.mat.needsUpdate = true })
            .catch(() => {})
        }
        continue
      }
      this.add(b)
    }
    for (const [slug, f] of this.bySlug) {
      if (!seen.has(slug)) { this.retire(f); this.bySlug.delete(slug) }
    }
  }

  private add(b: VillageBeingPos) {
    const scale = STAGE_SCALE[b.stage] ?? 1
    const h = FIG_H * scale
    const group = new THREE.Group()
    const mat = new THREE.MeshBasicMaterial({
      transparent: true, alphaTest: 0.15, side: THREE.DoubleSide,
      depthWrite: true, opacity: b.state === 'alive' ? 1 : 0.55,
    })
    mat.visible = false
    // the plane shows the whole canvas (drawing + sticker edge): 168:224
    const paper = new THREE.Mesh(new THREE.PlaneGeometry(h * (168 / 224), h), mat)
    paper.position.y = h / 2
    group.add(paper)
    const tag = nameTag(b.name)
    tag.position.y = h + 0.34
    group.add(tag)
    const mark = spark()
    mark.position.y = h + 0.72
    group.add(mark)
    const fig: Figure = {
      group, paper, mat, tag, mark, b,
      avatarKey: `${b.avatar?.c ?? 1}:${b.avatar?.p ?? 'ember'}`,
      h, bobPhase: 0, lastXZ: null, sensedAt: -1e9, frozen: null,
    }
    void rasterize(b.avatar?.c ?? 1, b.avatar?.p ?? 'ember')
      .then((tex) => { mat.map = tex; mat.visible = true; mat.needsUpdate = true })
      .catch(() => { mat.color.set(PALETTES[b.avatar?.p ?? 'ember']?.c1 ?? '#c46a3f'); mat.visible = true })
    this.scene.add(group)
    this.bySlug.set(b.slug, fig)
  }

  private retire(f: Figure) {
    this.scene.remove(f.group)
    f.paper.geometry.dispose(); f.mat.dispose()
    f.tag.material.map?.dispose(); f.tag.material.dispose()
    f.mark.material.map?.dispose(); f.mark.material.dispose()
  }

  // per frame: true clock position, billboard, bob, and the sensing pause
  update(dt: number, playerPos: { x: number; y: number; z: number }) {
    this.clock += dt
    for (const f of this.bySlug.values()) {
      const [ux, uz] = posOf(f.b, this.placeById, this.fetchedAtMs)
      const liveX = ux / UNITS_PER_BLOCK + MARGIN
      const liveZ = uz / UNITS_PER_BLOCK + MARGIN

      // sensing: near + off cooldown → stand still and listen for a spell
      const dx = playerPos.x - liveX, dz = playerPos.z - liveZ
      const near = Math.hypot(dx, dz) < SENSE_RADIUS
      const since = this.clock - f.sensedAt
      if (near && since > SENSE_COOLDOWN_S) {
        f.sensedAt = this.clock
        f.frozen = [f.group.position.x || liveX, f.group.position.z || liveZ]
      }
      const sensing = this.clock - f.sensedAt < SENSE_HOLD_S
      let x = liveX, z = liveZ
      if (sensing && f.frozen) {
        [x, z] = f.frozen
      } else if (f.frozen) {
        // the spell breaks: ease back onto the true clock
        const t = Math.min(1, (this.clock - f.sensedAt - SENSE_HOLD_S) / BLEND_S)
        x = f.frozen[0] + (liveX - f.frozen[0]) * t
        z = f.frozen[1] + (liveZ - f.frozen[1]) * t
        if (t >= 1) f.frozen = null
      }

      // walking bob — a paper figure's gait is a gentle seesaw
      const moved = f.lastXZ ? Math.hypot(x - f.lastXZ[0], z - f.lastXZ[1]) : 0
      const walking = moved / Math.max(dt, 1e-6) > 0.02 && !sensing
      if (walking) f.bobPhase += dt * 7
      const sway = walking ? Math.sin(f.bobPhase) * 0.055 : 0
      const lift = walking ? Math.abs(Math.sin(f.bobPhase)) * 0.05 : 0
      f.lastXZ = [x, z]

      // the sense-hop: one small jump the moment they feel it
      const hopT = this.clock - f.sensedAt
      const hop = hopT < 0.45 ? Math.sin((hopT / 0.45) * Math.PI) * 0.28 : 0
      const markMat = f.mark.material as THREE.SpriteMaterial
      markMat.opacity = sensing ? Math.min(1, since * 4) * (1 - Math.max(0, (since - SENSE_HOLD_S + 0.6)) / 0.6) : 0

      f.group.position.set(x, SURFACE + lift + hop, z)
      f.paper.rotation.z = sway
      // billboard around y only: the paper always shows its face
      f.group.rotation.y = Math.atan2(playerPos.x - x, playerPos.z - z)
    }
  }

  dispose() {
    for (const f of this.bySlug.values()) this.retire(f)
    this.bySlug.clear()
  }
}
