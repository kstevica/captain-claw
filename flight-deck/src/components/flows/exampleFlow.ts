// A guided, heavily-commented example flow for the Code view's "Load example"
// button. It compiles cleanly against the DSL parser, so users can read the
// comments and then hit "Validate & apply" to watch it become a real flow.
export const EXAMPLE_FLOW_DSL = `# ─────────────────────────────────────────────────────────────────────
#  EXAMPLE FLOW — "Smart photo triage"  (a guided tour of the DSL)
#
#  Lines starting with # are comments. A flow = a trigger + steps + an
#  output. Steps run top-to-bottom unless a branch jumps elsewhere.
#  {{...}} pulls values from the trigger or earlier steps.
# ─────────────────────────────────────────────────────────────────────

flow "Smart photo triage"
description "Recognize faces in inbound photos; greet, ask, or describe."

# Fire on ANY WhatsApp photo (captionless glasses photos included).
trigger whatsapp when has image

# 1) Identify faces. Runs INSIDE Flight Deck (on: fd) — no agent turn —
#    reading the FD-local copy of the photo. face_identify returns JSON
#    whose fields become {{steps.identify.<field>}}: confident, name, count.
step identify:
  tool on fd
  tool: face_identify
  arg image: {{trigger.fd_image_path}}

# 2) Route on the result. First matching condition wins; else → fallback.
step route:
  branch
  if {{steps.identify.confident}} == true -> greet     # a known person
  elif {{steps.identify.count}} == 0 -> describe        # no face at all
  else -> ask_who                                       # a face, but unknown

# 3a) Known person → greet by name, then end the flow.
step greet:
  emit "👋 Hey {{steps.identify.name}}! Good to see you."
  stop

# 3b) Unknown face → PAUSE and ask who it is. The reply becomes
#     {{steps.ask_who.output}} and the flow resumes here.
step ask_who:
  input
  prompt: "I don't recognize this person — who is it? (reply with a name)"

# 4) Acknowledge, then end.
step ack:
  emit "Thanks! To save the face, send the photo with: remember this is {{steps.ask_who.output}}"
  stop

# 3c) No face → describe the scene with the vision model (a different,
#     vision-capable agent), then end. This is the last step, so its
#     output is delivered as the flow's reply.
step describe:
  vision on capability:vision
  prompt: "Describe what's in this photo in one or two sentences."
  image: {{trigger.fd_image_path}}

# Where the final/emitted messages go. \`same\` = reply on the origin channel.
output -> same
`
