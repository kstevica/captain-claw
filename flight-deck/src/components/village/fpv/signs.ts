// Signs in the grass (FPV plan Phase 3): each planted note stands in the
// world as a little wooden post with a paper slip. The paper billboards
// toward the ghost (y-axis only) so it always reads as a note; the words
// themselves surface in the HUD when you stand close — paper at this size
// holds no legible type.

import * as THREE from 'three'
import type { VillageNote } from '../../../services/beings'
import { MARGIN, SURFACE } from './worldgen'

const UNITS_PER_BLOCK = 5
export const SIGN_READ_RADIUS = 2.6   // blocks: close enough to read

// one shared paper texture — a cream slip with faint written lines
function paperTexture(): THREE.CanvasTexture {
  const cv = document.createElement('canvas')
  cv.width = 64; cv.height = 48
  const ctx = cv.getContext('2d')!
  ctx.fillStyle = '#f2ead2'
  ctx.fillRect(0, 0, 64, 48)
  ctx.strokeStyle = 'rgba(120,94,58,.85)'
  ctx.lineWidth = 2
  ctx.strokeRect(1, 1, 62, 46)
  ctx.strokeStyle = 'rgba(90,70,50,.5)'
  ctx.lineWidth = 1.5
  for (let y = 12; y < 42; y += 8) {
    ctx.beginPath()
    ctx.moveTo(8, y)
    // a wavering hand — nobody writes straight in the grass
    for (let x = 8; x <= 50 - (y % 16); x += 6) {
      ctx.lineTo(x, y + Math.sin(x * 1.7 + y) * 1.2)
    }
    ctx.stroke()
  }
  const tex = new THREE.CanvasTexture(cv)
  tex.colorSpace = THREE.SRGBColorSpace
  return tex
}

interface Sign {
  group: THREE.Group
  paper: THREE.Mesh
  note: VillageNote
}

export class Signs {
  private scene: THREE.Scene
  private byId = new Map<string, Sign>()
  private postGeo = new THREE.BoxGeometry(0.09, 1.0, 0.09)
  private postMat = new THREE.MeshLambertMaterial({ color: 0x6b5138 })
  private paperGeo = new THREE.PlaneGeometry(0.62, 0.46)
  private paperMat: THREE.MeshBasicMaterial

  constructor(scene: THREE.Scene) {
    this.scene = scene
    this.paperMat = new THREE.MeshBasicMaterial({
      map: paperTexture(), side: THREE.DoubleSide })
  }

  sync(notes: VillageNote[]) {
    const seen = new Set<string>()
    for (const n of notes) {
      seen.add(n.id)
      const have = this.byId.get(n.id)
      if (have) { have.note = n; continue }
      const group = new THREE.Group()
      const post = new THREE.Mesh(this.postGeo, this.postMat)
      post.position.y = 0.5
      const paper = new THREE.Mesh(this.paperGeo, this.paperMat)
      paper.position.y = 1.06
      group.add(post, paper)
      group.position.set(n.x / UNITS_PER_BLOCK + MARGIN, SURFACE,
                         n.y / UNITS_PER_BLOCK + MARGIN)
      this.scene.add(group)
      this.byId.set(n.id, { group, paper, note: n })
    }
    for (const [id, s] of this.byId) {
      if (!seen.has(id)) { this.scene.remove(s.group); this.byId.delete(id) }
    }
  }

  // per frame: papers turn to face the ghost
  update(playerPos: { x: number; z: number }) {
    for (const s of this.byId.values()) {
      s.paper.rotation.y = Math.atan2(playerPos.x - s.group.position.x,
                                      playerPos.z - s.group.position.z)
    }
  }

  // the sign underfoot (for the HUD reader) — nearest within reach
  nearest(playerPos: { x: number; z: number }): VillageNote | null {
    let best: VillageNote | null = null
    let bestD = SIGN_READ_RADIUS
    for (const s of this.byId.values()) {
      const d = Math.hypot(playerPos.x - s.group.position.x,
                           playerPos.z - s.group.position.z)
      if (d < bestD) { bestD = d; best = s.note }
    }
    return best
  }

  dispose() {
    for (const s of this.byId.values()) this.scene.remove(s.group)
    this.byId.clear()
    this.postGeo.dispose(); this.postMat.dispose()
    this.paperGeo.dispose()
    this.paperMat.map?.dispose(); this.paperMat.dispose()
  }
}
