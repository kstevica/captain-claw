// Reading stands (FPV plan Phase 4): a small angled lectern stands inside
// each building whose Iskre keep work there — a post, a slanted paper
// board, a soft glow so it reads as "come read here". Step within reach
// and the engine offers a 'press R to read' prompt; R opens the same
// per-iskra file browser the 2D map uses, in first person.

import * as THREE from 'three'
import type { Lectern } from './worldgen'
import { SURFACE } from './worldgen'

export const LECTERN_RADIUS = 2.4   // blocks: close enough to read

export class Lecterns {
  private scene: THREE.Scene
  private items: { l: Lectern; group: THREE.Group }[] = []
  private postGeo = new THREE.BoxGeometry(0.12, 1.05, 0.12)
  private postMat = new THREE.MeshLambertMaterial({ color: 0x6b5138 })
  private boardGeo = new THREE.BoxGeometry(0.6, 0.42, 0.06)
  private boardMat = new THREE.MeshLambertMaterial({ color: 0xe8dcc0 })
  private glowGeo = new THREE.PlaneGeometry(0.66, 0.48)
  private glowMat = new THREE.MeshBasicMaterial({
    color: 0xffe6ad, transparent: true, opacity: 0.32,
    depthWrite: false, side: THREE.DoubleSide })

  constructor(scene: THREE.Scene, lecterns: Lectern[]) {
    this.scene = scene
    for (const l of lecterns) {
      const group = new THREE.Group()
      const post = new THREE.Mesh(this.postGeo, this.postMat)
      post.position.y = 0.52
      const board = new THREE.Mesh(this.boardGeo, this.boardMat)
      board.position.set(0, 1.0, 0.06)
      board.rotation.x = -0.5           // tilt the reading face up
      const glow = new THREE.Mesh(this.glowGeo, this.glowMat)
      glow.position.set(0, 1.0, 0.09)
      glow.rotation.x = -0.5
      group.add(post, board, glow)
      group.position.set(l.bx + 0.5, SURFACE, l.bz + 0.5)
      this.scene.add(group)
      this.items.push({ l, group })
    }
  }

  // gentle bob so the stand catches the eye, and it faces the ghost
  update(t: number, playerPos: { x: number; z: number }) {
    for (const { group } of this.items) {
      group.position.y = SURFACE + Math.sin(t * 1.6 + group.position.x) * 0.03
      group.rotation.y = Math.atan2(playerPos.x - group.position.x,
                                    playerPos.z - group.position.z)
    }
  }

  // the stand within reach (for the prompt + the R key)
  nearest(playerPos: { x: number; z: number }): Lectern | null {
    let best: Lectern | null = null
    let bestD = LECTERN_RADIUS
    for (const { l, group } of this.items) {
      const d = Math.hypot(playerPos.x - group.position.x,
                           playerPos.z - group.position.z)
      if (d < bestD) { bestD = d; best = l }
    }
    return best
  }

  dispose() {
    for (const { group } of this.items) this.scene.remove(group)
    this.items = []
    this.postGeo.dispose(); this.postMat.dispose()
    this.boardGeo.dispose(); this.boardMat.dispose()
    this.glowGeo.dispose(); this.glowMat.dispose()
  }
}
