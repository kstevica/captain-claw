// The other ghosts (FPV plan Phase 5): the parent and public visitors roam
// one village together and see each other as soft, translucent figures —
// not the paper-cutout Iskre, but wisps of presence, each wearing its
// identity pill (violet "parent", amber visitor name). Positions arrive on
// a ~2s heartbeat, so each figure LERPS toward its latest spot (unlike the
// Iskre, whose motion is a deterministic clock function). A ghost that
// stops heartbeating fades and is retired.

import * as THREE from 'three'
import type { GhostPresence } from '../../../services/beings'
import { MARGIN, SURFACE } from './worldgen'

const UNITS_PER_BLOCK = 5

// a soft spectral figure — a rounded sheet with a wavy hem and two eyes
function ghostTexture(): THREE.CanvasTexture {
  const cv = document.createElement('canvas')
  cv.width = 96; cv.height = 128
  const ctx = cv.getContext('2d')!
  ctx.clearRect(0, 0, 96, 128)
  ctx.fillStyle = '#ffffff'
  ctx.beginPath()
  // domed head + shoulders
  ctx.moveTo(14, 118)
  ctx.lineTo(14, 56)
  ctx.arc(48, 56, 34, Math.PI, 0)          // the head dome
  ctx.lineTo(82, 118)
  // a wavy hem across the bottom
  for (let i = 0; i < 4; i++) {
    const x0 = 82 - i * 17
    ctx.quadraticCurveTo(x0 - 8.5, 106, x0 - 17, 118)
  }
  ctx.closePath()
  ctx.fill()
  // eyes (punched darker so a tint still reads a face)
  ctx.globalCompositeOperation = 'destination-out'
  ctx.beginPath(); ctx.arc(38, 52, 5, 0, 7); ctx.fill()
  ctx.beginPath(); ctx.arc(58, 52, 5, 0, 7); ctx.fill()
  const tex = new THREE.CanvasTexture(cv)
  tex.colorSpace = THREE.SRGBColorSpace
  return tex
}

function pillTexture(text: string, tint: string): THREE.CanvasTexture {
  const cv = document.createElement('canvas')
  cv.width = 256; cv.height = 64
  const ctx = cv.getContext('2d')!
  ctx.font = '600 26px -apple-system, system-ui, sans-serif'
  const w = Math.min(240, ctx.measureText(text).width + 30)
  const x0 = (256 - w) / 2
  ctx.fillStyle = 'rgba(23,20,16,0.88)'
  ctx.beginPath(); ctx.roundRect(x0, 12, w, 40, 20); ctx.fill()
  ctx.lineWidth = 2; ctx.strokeStyle = tint
  ctx.beginPath(); ctx.roundRect(x0, 12, w, 40, 20); ctx.stroke()
  ctx.fillStyle = tint
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle'
  ctx.fillText(text, 128, 33)
  const tex = new THREE.CanvasTexture(cv)
  tex.colorSpace = THREE.SRGBColorSpace
  return tex
}

const TINT = { parent: '#c4b5fd', visitor: '#fcd9a0' } as const
const BODY = { parent: 0xb7a6ee, visitor: 0xf0d29a } as const
const H = 1.9   // ghost height in blocks

interface Ghost {
  group: THREE.Group
  body: THREE.Mesh
  bodyMat: THREE.MeshBasicMaterial
  pill: THREE.Sprite
  target: THREE.Vector2   // block-space target (x, z)
  cur: THREE.Vector2      // lerped
  g: GhostPresence
  phase: number
}

export class Ghosts {
  private scene: THREE.Scene
  private byId = new Map<string, Ghost>()
  private bodyGeo = new THREE.PlaneGeometry(H * (0.75), H)
  private tex = ghostTexture()

  constructor(scene: THREE.Scene) {
    this.scene = scene
  }

  private toBlock(xy: [number, number]): [number, number] {
    return [xy[0] / UNITS_PER_BLOCK + MARGIN, xy[1] / UNITS_PER_BLOCK + MARGIN]
  }

  sync(ghosts: GhostPresence[]) {
    const seen = new Set<string>()
    for (const g of ghosts) {
      seen.add(g.id)
      const [bx, bz] = this.toBlock(g.xy)
      const have = this.byId.get(g.id)
      if (have) {
        have.g = g
        have.target.set(bx, bz)
        // a changed name/kind → repaint the pill
        const label = g.kind === 'parent' ? 'parent' : g.name
        if ((have.pill.userData.label as string) !== label) {
          have.pill.material.map?.dispose()
          const t = pillTexture(label, TINT[g.kind])
          ;(have.pill.material as THREE.SpriteMaterial).map = t
          have.pill.material.needsUpdate = true
          have.pill.userData.label = label
        }
        continue
      }
      const group = new THREE.Group()
      const bodyMat = new THREE.MeshBasicMaterial({
        map: this.tex, transparent: true, opacity: 0.5,
        color: BODY[g.kind], depthWrite: false, side: THREE.DoubleSide })
      const body = new THREE.Mesh(this.bodyGeo, bodyMat)
      body.position.y = H / 2 + 0.15
      const label = g.kind === 'parent' ? 'parent' : g.name
      const pill = new THREE.Sprite(new THREE.SpriteMaterial({
        map: pillTexture(label, TINT[g.kind]), transparent: true,
        depthWrite: false }))
      pill.scale.set(1.7, 0.42, 1)
      pill.position.y = H + 0.5
      pill.userData.label = label
      group.add(body, pill)
      group.position.set(bx, SURFACE, bz)
      this.scene.add(group)
      this.byId.set(g.id, {
        group, body, bodyMat, pill, g,
        target: new THREE.Vector2(bx, bz),
        cur: new THREE.Vector2(bx, bz),
        phase: 0,
      })
    }
    for (const [id, gh] of this.byId) {
      if (!seen.has(id)) { this.retire(gh); this.byId.delete(id) }
    }
  }

  private retire(gh: Ghost) {
    this.scene.remove(gh.group)
    gh.bodyMat.dispose()
    gh.pill.material.map?.dispose()
    gh.pill.material.dispose()
  }

  update(dt: number, playerPos: { x: number; z: number }) {
    const k = Math.min(1, dt * 3)       // lerp catch-up (~2s to arrive)
    for (const gh of this.byId.values()) {
      gh.cur.lerp(gh.target, k)
      gh.phase += dt
      const bob = Math.sin(gh.phase * 1.5) * 0.08
      gh.group.position.set(gh.cur.x, SURFACE + 0.1 + bob, gh.cur.y)
      // billboard the sheet toward the local ghost, y-axis only (the pill
      // is a Sprite and faces the camera on its own)
      gh.body.rotation.y = Math.atan2(playerPos.x - gh.cur.x,
                                      playerPos.z - gh.cur.y)
    }
  }

  dispose() {
    for (const gh of this.byId.values()) this.retire(gh)
    this.byId.clear()
    this.bodyGeo.dispose()
    this.tex.dispose()
  }
}
