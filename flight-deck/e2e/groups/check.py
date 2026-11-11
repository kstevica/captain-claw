"""Agent groups in the kiosk, on the throwaway deck run.sh boots: the real bundle, real /fd/settings.

The sign-in-off check stubs the deck's answers: that deck keeps its groups in the browser.
"""
import json
import os
import sys
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

BASE = os.environ["BASE"]
SHOTS = Path(os.environ["E2E_RUN_DIR"]) / "shots"
SHOTS.mkdir(parents=True, exist_ok=True)
PASSWORD = "correct-horse-battery"
results: list[tuple[str, bool, str]] = []
HEAD = ".group\\/head button[aria-expanded]"


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(("PASS " if ok else "FAIL ") + name + (f"  — {detail}" if detail else ""), flush=True)


def sign_in(page, email):
    page.goto(BASE)
    page.fill("input[type=email]", email)
    page.fill("input[type=password]", PASSWORD)
    page.click("button[type=submit]")


def close_guide(page):
    page.wait_for_timeout(800)
    guide = page.get_by_role("dialog", name="Getting started")
    if guide.count():
        page.keyboard.press("Escape")
        page.wait_for_timeout(200)


def section(page, name):
    return page.locator(f'section[data-section="{name}"]')


def names_in(page, name):
    return section(page, name).locator("li span.font-medium").all_text_contents()


def order(page):
    return page.locator("section[data-section]").evaluate_all("els => els.map(e => e.dataset.section)")


def server_groups(page):
    raw = page.evaluate(
        """async () => {
            const token = await fetch('/fd/auth/refresh', {method:'POST', credentials:'include'}).then(r => r.json()).then(d => d.access_token)
            const s = await fetch('/fd/settings', {headers:{Authorization:'Bearer '+token}, credentials:'include'}).then(r => r.json())
            return s['fd:agent-groups'] || '[]'
        }"""
    )
    return json.loads(raw)


def row(page, name):
    return page.locator("aside li", has_text=name).first


def drag_to(page, source, target, y_frac=0.5):
    s, t = source.bounding_box(), target.bounding_box()
    page.mouse.move(s["x"] + s["width"] / 2, s["y"] + s["height"] / 2)
    page.mouse.down()
    page.mouse.move(s["x"] + s["width"] / 2 + 5, s["y"] + s["height"] / 2 + 5, steps=3)
    page.wait_for_timeout(120)
    page.mouse.move(t["x"] + t["width"] / 2, t["y"] + t["height"] * y_frac, steps=12)
    page.wait_for_timeout(80)
    page.mouse.move(t["x"] + t["width"] / 2 + 1, t["y"] + t["height"] * y_frac, steps=1)
    page.wait_for_timeout(220)


def shot(page, name):
    page.screenshot(path=str(SHOTS / f"{name}.png"))


def kiosk_user(browser):
    ctx = browser.new_context(viewport={"width": 1280, "height": 860})
    ctx.add_init_script("try{localStorage.setItem('fd:theme','dark')}catch{}")
    page = ctx.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    sign_in(page, "maya@example.com")
    page.wait_for_selector("aside li")
    close_guide(page)
    page.evaluate(
        """async () => {
            const token = await fetch('/fd/auth/refresh', {method:'POST', credentials:'include'}).then(r => r.json()).then(d => d.access_token)
            const h = {Authorization:'Bearer '+token, 'Content-Type':'application/json'}
            await fetch('/fd/settings', {method:'PUT', headers:h, credentials:'include', body: JSON.stringify({settings:{'fd:agent-groups': '[]'}})})
            localStorage.removeItem('fd:agent-groups-collapsed')
        }"""
    )
    page.reload()
    page.wait_for_selector("aside li")
    close_guide(page)

    # ── No groups: the plain list ──
    check("no groups: no section headers", page.locator("section[data-section]").count() == 0)
    check("no groups: all 8 agents listed", page.locator("aside li").count() == 8)
    check("no groups: a New group row", page.get_by_role("button", name="New group").count() >= 1)
    shot(page, "01-flat")

    # ── New group: name + ticks ──
    page.get_by_title("New group").click()
    dialog = page.get_by_role("dialog", name="New group")
    expect(dialog).to_be_visible()
    check("dialog: name focused", page.evaluate("document.activeElement.getAttribute('aria-label')") == "Group name")
    check("dialog: can't create without a name", dialog.get_by_role("button", name="Create group").is_disabled())
    dialog.get_by_label("Group name").fill("Research")
    dialog.get_by_role("checkbox", name="Research Assistant").click()
    dialog.get_by_role("checkbox", name="Market Watch").click()
    dialog.get_by_role("checkbox", name="Report Writer").click()
    shot(page, "02-new-group-dialog")
    dialog.get_by_role("button", name="Create group").click()
    expect(section(page, "Research")).to_be_visible()
    check("created: its agents in it", sorted(names_in(page, "Research")) == ["Market Watch", "Report Writer", "Research Assistant"], str(names_in(page, "Research")))
    check("created: the rest in Other agents", len(names_in(page, "Other agents")) == 5)
    page.wait_for_timeout(600)
    stored = server_groups(page)
    check("saved to the server", len(stored) == 1 and stored[0]["name"] == "Research" and len(stored[0]["agentIds"]) == 3, json.dumps(stored)[:120])
    shot(page, "03-one-group")

    # ── Reload: from the server ──
    page.reload()
    page.wait_for_selector('section[data-section="Research"]')
    close_guide(page)
    check("reload: still grouped", len(names_in(page, "Research")) == 3)

    # ── Drag an agent onto a group ──
    drag_to(page, row(page, "Legal Reader"), section(page, "Research").locator(HEAD))
    shot(page, "04-dragging-over-group")
    check("dragging: Drop-here row shows", page.get_by_role("button", name="Drop here for a new group").count() == 1)
    page.mouse.up()
    page.wait_for_timeout(150)
    shot(page, "05-landed")
    check("drag: agent landed in the group", "Legal Reader" in names_in(page, "Research"), str(names_in(page, "Research")))
    check("drag: left Other agents", "Legal Reader" not in names_in(page, "Other agents"))
    check("drag: nothing stuck dimmed", page.locator('aside li[style*="opacity"]').count() == 0)

    # ── Drag onto New group: dialog, agent already ticked ──
    drag_to(page, row(page, "Legal Reader"), page.locator("aside").get_by_role("button", name="New group").last)
    page.mouse.up()
    dialog = page.get_by_role("dialog", name="New group")
    expect(dialog).to_be_visible()
    ticked = dialog.get_by_role("checkbox", name="Legal Reader")
    check("drop on New group: agent ticked", ticked.get_attribute("aria-checked") == "true")
    check("drop on New group: says it moves", "Moves here from Research" in ticked.inner_text(), ticked.inner_text())
    dialog.get_by_label("Group name").fill("research")
    check("same name refused", dialog.get_by_role("button", name="Create group").is_disabled() and dialog.get_by_text("already have a group").count() == 1)
    dialog.get_by_label("Group name").fill("Legal")
    dialog.get_by_role("radio", name="emerald").click()
    shot(page, "06-drop-new-group")
    dialog.get_by_label("Group name").press("Enter")
    expect(section(page, "Legal")).to_be_visible()
    check("moved out of the old group", "Legal Reader" not in names_in(page, "Research") and names_in(page, "Legal") == ["Legal Reader"])

    # ── Drag out, back to Other agents ──
    drag_to(page, row(page, "Market Watch"), section(page, "Other agents").locator(HEAD))
    page.mouse.up()
    page.wait_for_timeout(150)
    check("drag to Other agents takes it out", "Market Watch" in names_in(page, "Other agents"))

    # ── ⋯ → Group: picker ──
    row(page, "Market Watch").hover()
    page.get_by_role("button", name="More for Market Watch").click()
    menu = page.get_by_role("dialog", name="Market Watch")
    expect(menu).to_be_visible()
    page.wait_for_timeout(350)
    shot(page, "07-agent-menu")
    menu.get_by_role("button", name="Group").click()
    picker = page.get_by_role("dialog", name="Group for Market Watch")
    expect(picker).to_be_visible()
    check("picker: no group ticked first", picker.get_by_role("radio", name="No group").get_attribute("aria-checked") == "true")
    page.wait_for_timeout(350)
    shot(page, "08-picker")
    picker.get_by_role("radio", name="Research").click()
    expect(picker).to_be_hidden()
    check("picker: moved", "Market Watch" in names_in(page, "Research"))

    # ── Picker: new group by name ──
    row(page, "Travel Desk").hover()
    page.get_by_role("button", name="More for Travel Desk").click()
    page.get_by_role("dialog", name="Travel Desk").get_by_role("button", name="Group").click()
    picker = page.get_by_role("dialog", name="Group for Travel Desk")
    picker.get_by_label("New group name").fill("Home")
    picker.get_by_label("New group name").press("Enter")
    expect(section(page, "Home")).to_be_visible()
    check("picker: new group + agent", names_in(page, "Home") == ["Travel Desk"])

    # ── Open a chat, then collapse: the open chat's row stays ──
    row(page, "Research Assistant").locator("button").first.click()
    page.wait_for_timeout(600)
    section(page, "Research").locator(HEAD).click()
    page.wait_for_timeout(350)
    head = section(page, "Research").locator(HEAD)
    check("collapse: aria-expanded false", head.get_attribute("aria-expanded") == "false")
    visible = [e.inner_text() for e in section(page, "Research").locator("li span.font-medium").all() if e.is_visible()]
    check("collapse: active chat's row stays in view", "Research Assistant" in visible, str(visible))
    report = section(page, "Research").locator("li", has_text="Report Writer").first
    box = report.bounding_box()
    check("collapse: the others have no height", box is None or box["height"] == 0 or not report.is_visible() or report.evaluate("e => !!e.closest('[inert]')"))
    check("collapse: others not focusable (inert)", report.evaluate("e => !!e.closest('[inert]')"))
    shot(page, "09-collapsed")

    # ── Collapse survives reload, this device only ──
    page.reload()
    page.wait_for_selector('section[data-section="Research"]')
    close_guide(page)
    check("collapse: survives reload", section(page, "Research").locator(HEAD).get_attribute("aria-expanded") == "false")
    check("collapse: not written to the server", "collapsed" not in json.dumps(server_groups(page)))

    # ── Search: reaches inside folded groups, and by group name ──
    search = page.get_by_placeholder("Search agents…")
    search.fill("report")
    page.wait_for_timeout(250)
    check("search: inside a folded group", section(page, "Research").locator("li", has_text="Report Writer").first.is_visible())
    check("search: other sections gone", order(page) == ["Research"], str(order(page)))
    search.fill("legal")
    page.wait_for_timeout(200)
    check("search: by group name", names_in(page, "Legal") == ["Legal Reader"])
    search.fill("zzzz")
    page.wait_for_timeout(200)
    check("search: says nothing matches", page.get_by_text("No agents match your search.").is_visible())
    search.fill("")
    section(page, "Research").locator(HEAD).click()
    page.wait_for_timeout(300)

    # ── Reorder: drag a group's header ──
    before = order(page)
    drag_to(page, section(page, "Home").locator(HEAD), section(page, "Research").locator(HEAD), y_frac=0.1)
    shot(page, "10-reordering")
    page.mouse.up()
    page.wait_for_timeout(200)
    after = order(page)
    check("reorder: dragged above", after.index("Home") < after.index("Research"), f"{before} → {after}")
    check("reorder: Other agents stays last", after[-1] == "Other agents", str(after))
    page.wait_for_timeout(500)
    check("reorder: saved", [g["name"] for g in server_groups(page)][0] == "Home", str([g["name"] for g in server_groups(page)]))

    # ── Edit: rename, colour, members, order ──
    section(page, "Research").locator(".group\\/head").hover()
    page.get_by_role("button", name="Edit group Research").click()
    dialog = page.get_by_role("dialog", name="Edit group Research")
    expect(dialog).to_be_visible()
    dialog.get_by_label("Group name").fill("Deep research")
    dialog.get_by_role("radio", name="amber").click()
    dialog.get_by_role("checkbox", name="Grant Scout").click()
    dialog.get_by_role("checkbox", name="Report Writer").click()
    page.wait_for_timeout(350)
    shot(page, "11-edit-dialog")
    dialog.get_by_role("button", name="Save").click()
    expect(section(page, "Deep research")).to_be_visible()
    names = names_in(page, "Deep research")
    check("edit: ticks applied", "Grant Scout" in names and "Report Writer" not in names, str(names))
    check("edit: unticked goes back", "Report Writer" in names_in(page, "Other agents"))
    page.wait_for_timeout(500)
    stored = {g["name"]: g for g in server_groups(page)}
    check("edit: colour saved", stored["Deep research"]["color"] == "amber")
    shot(page, "12-three-groups")

    # ── A new reply on a folded header ──
    row(page, "Inbox Helper").locator("button").first.click()
    page.wait_for_timeout(700)
    box = page.get_by_placeholder("Ask anything").first
    box.fill("Anything urgent overnight?")
    box.press("Enter")
    row(page, "Research Assistant").locator("button").first.click()
    page.wait_for_timeout(2500)
    section(page, "Other agents").locator(HEAD).click()
    page.wait_for_timeout(350)
    # The mock takes the message and never finishes: the agent stays at work.
    signal = section(page, "Other agents").locator(".group\\/head").locator('[title="An agent here is working"]')
    check("folded header shows an agent at work", signal.count() == 1)
    shot(page, "12b-new-reply-folded")
    section(page, "Other agents").locator(HEAD).click()
    page.wait_for_timeout(300)

    # ── Rail ──
    page.get_by_title("Hide agents").click()
    page.wait_for_timeout(200)
    marks = page.locator("aside button[aria-expanded]")
    check("rail: a mark per section", marks.count() == 4, str(marks.count()))
    shot(page, "13-rail")
    tiles_before = page.locator("aside button[title^='Chat with'], aside button[aria-disabled]").count()
    page.get_by_role("button", name="Deep research").click()
    page.wait_for_timeout(200)
    tiles_after = page.locator("aside button[title^='Chat with'], aside button[aria-disabled]").count()
    check("rail: folding hides its agents", tiles_after < tiles_before, f"{tiles_before} → {tiles_after}")
    shot(page, "14-rail-folded")
    page.get_by_role("button", name="Deep research").click()
    page.get_by_title("Show agents").click()
    page.wait_for_timeout(200)

    # ── Another device changes the groups: coming back to this tab shows them ──
    page.evaluate(
        """async () => {
            const token = await fetch('/fd/auth/refresh', {method:'POST', credentials:'include'}).then(r => r.json()).then(d => d.access_token)
            const h = {Authorization:'Bearer '+token, 'Content-Type':'application/json'}
            const s = await fetch('/fd/settings', {headers:h, credentials:'include'}).then(r => r.json())
            const g = JSON.parse(s['fd:agent-groups'])
            g.find(x => x.name === 'Home').name = 'Personal'
            await fetch('/fd/settings', {method:'PUT', headers:h, credentials:'include', body: JSON.stringify({settings:{'fd:agent-groups': JSON.stringify(g)}})})
        }"""
    )
    page.wait_for_timeout(10500)
    page.evaluate("window.dispatchEvent(new Event('focus'))")
    page.wait_for_timeout(700)
    check("back on the tab: another device's change shows", section(page, "Personal").count() == 1, str(order(page)))

    # ── Delete: agents stay ──
    section(page, "Legal").locator(".group\\/head").hover()
    page.get_by_role("button", name="Edit group Legal").click()
    dialog = page.get_by_role("dialog", name="Edit group Legal")
    dialog.get_by_role("button", name="Delete").click()
    check("delete: asks first, says agents stay", dialog.get_by_text("Its agents stay").is_visible())
    page.wait_for_timeout(350)
    shot(page, "15-delete")
    dialog.get_by_role("button", name="Delete").click()
    expect(section(page, "Legal")).to_have_count(0)
    check("delete: agent back in Other agents", "Legal Reader" in names_in(page, "Other agents"))

    # ── Remove an agent for good: out of its group ──
    row(page, "Travel Desk").hover()
    page.get_by_role("button", name="More for Travel Desk").click()
    page.get_by_role("dialog", name="Travel Desk").get_by_role("button", name="Remove agent").click()
    remove = page.get_by_role("dialog").last
    typed = remove.locator("input")
    if typed.count():
        typed.first.fill("Travel Desk")
    remove.get_by_role("button", name="Remove").last.click()
    page.wait_for_timeout(2100)
    ids = [i for g in server_groups(page) for i in g["agentIds"]]
    check("remove: agent gone from its group", "proc-travel-desk" not in ids, str(ids))

    check("no page errors", not errors, "; ".join(errors)[:300])

    # ── Light theme ──
    page.get_by_role("button", name="Settings").click()
    page.wait_for_timeout(300)
    page.get_by_role("switch", name="Dark theme").click()
    page.keyboard.press("Escape")
    page.wait_for_timeout(400)
    shot(page, "16-light")
    section(page, "Deep research").locator(".group\\/head").hover()
    page.get_by_role("button", name="Edit group Deep research").click()
    page.wait_for_timeout(300)
    shot(page, "17-light-dialog")
    page.keyboard.press("Escape")
    page.wait_for_timeout(200)
    page.get_by_role("button", name="Settings").click()
    page.wait_for_timeout(300)
    page.get_by_role("switch", name="Dark theme").click()
    page.keyboard.press("Escape")
    page.wait_for_timeout(500)
    ctx.close()


def phone(browser):
    ctx = browser.new_context(viewport={"width": 390, "height": 844}, has_touch=True, is_mobile=True)
    ctx.add_init_script("try{localStorage.setItem('fd:theme','dark')}catch{}")
    page = ctx.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    sign_in(page, "maya@example.com")
    page.wait_for_selector("section[data-section]")
    close_guide(page)
    shot(page, "20-phone-list")
    check("phone: groups from the server", section(page, "Deep research").count() == 1)
    check("phone: no drag on rows", page.locator("li[draggable=true]").count() == 0)

    section(page, "Other agents").locator(HEAD).tap()
    page.wait_for_timeout(300)
    shot(page, "21-phone-collapsed")
    section(page, "Other agents").locator(HEAD).tap()
    page.wait_for_timeout(300)

    # An agent's ⋯ → Group
    page.get_by_role("button", name="More for Home Admin").tap()
    menu = page.get_by_role("dialog", name="Home Admin")
    expect(menu).to_be_visible()
    menu.get_by_role("button", name="Group").scroll_into_view_if_needed()
    menu.get_by_role("button", name="Group").tap()
    picker = page.get_by_role("dialog", name="Group for Home Admin")
    expect(picker).to_be_visible()
    page.wait_for_timeout(350)
    shot(page, "22-phone-picker")
    picker.get_by_role("radio", name="Personal").tap()
    page.wait_for_timeout(300)
    check("phone: picker moved it", "Home Admin" in names_in(page, "Personal"), str(names_in(page, "Personal")))

    # New group from the header, with ticks
    page.get_by_role("button", name="New group").first.tap()
    dialog = page.get_by_role("dialog", name="New group")
    expect(dialog).to_be_visible()
    page.wait_for_timeout(350)
    dialog.get_by_label("Group name").fill("Watchers")
    dialog.get_by_role("checkbox", name="Legal Reader").tap()
    shot(page, "23-phone-dialog")
    dialog.get_by_role("button", name="Create group").tap()
    expect(section(page, "Watchers")).to_be_visible()
    check("phone: new group", names_in(page, "Watchers") == ["Legal Reader"])

    # Order from the group's sheet
    page.get_by_role("button", name="Edit group Watchers").tap()
    dialog = page.get_by_role("dialog", name="Edit group Watchers")
    page.wait_for_timeout(350)
    before = order(page)
    dialog.get_by_role("button", name="Higher in the list").tap()
    page.wait_for_timeout(100)
    dialog.get_by_role("button", name="Save").tap()
    page.wait_for_timeout(200)
    after = order(page)
    check("phone: order changed from the sheet", after.index("Watchers") < before.index("Watchers"), f"{before} → {after}")
    width = page.evaluate("document.documentElement.scrollWidth")
    check("phone: no sideways scroll", width <= 390, str(width))
    shot(page, "24-phone-final")
    check("phone: no page errors", not errors, "; ".join(errors)[:300])
    ctx.close()


def put_groups(page, groups):
    page.evaluate(
        """async (groups) => {
            const token = await fetch('/fd/auth/refresh', {method:'POST', credentials:'include'}).then(r => r.json()).then(d => d.access_token)
            const h = {Authorization:'Bearer '+token, 'Content-Type':'application/json'}
            await fetch('/fd/settings', {method:'PUT', headers:h, credentials:'include', body: JSON.stringify({settings:{'fd:agent-groups': JSON.stringify(groups)}})})
            localStorage.removeItem('fd:agent-groups-collapsed')
        }""",
        groups,
    )


def pressed(page, name):
    return page.get_by_role("button", name=name, exact=True).get_attribute("aria-pressed") == "true"


def pins(browser):
    ctx = browser.new_context(viewport={"width": 1280, "height": 860})
    ctx.add_init_script("try{localStorage.setItem('fd:theme','dark')}catch{}")
    page = ctx.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    sign_in(page, "maya@example.com")
    page.wait_for_selector("aside li")
    close_guide(page)
    put_groups(page, [
        {"id": "g-a", "name": "Alpha", "color": "violet", "type": "group", "agentIds": ["proc-research-assistant", "proc-market-watch", "proc-report-writer"]},
        {"id": "g-b", "name": "Bravo", "color": "blue", "type": "group", "agentIds": ["proc-inbox-helper", "proc-grant-scout"]},
        {"id": "g-c", "name": "Charlie", "color": "emerald", "type": "group", "agentIds": ["proc-legal-reader"]},
    ])
    page.reload()
    page.wait_for_selector('section[data-section="Alpha"]')
    close_guide(page)
    head = lambda name: section(page, name).locator(".group\\/head")
    check("pins: start in their order", order(page)[:3] == ["Alpha", "Bravo", "Charlie"], str(order(page)))

    # ── One group to the top ──
    head("Charlie").hover()
    page.get_by_role("button", name="Pin group Charlie", exact=True).click()
    page.wait_for_timeout(150)
    shot(page, "40-group-pinned")
    check("pin group: goes first", order(page)[:3] == ["Charlie", "Alpha", "Bravo"], str(order(page)))
    check("pin group: says so", pressed(page, "Unpin group Charlie"))
    page.wait_for_timeout(500)
    stored = server_groups(page)
    check("pin group: saved, the list keeps its order", [g["name"] for g in stored] == ["Alpha", "Bravo", "Charlie"] and [g["name"] for g in stored if g.get("pinned")] == ["Charlie"], json.dumps(stored)[:200])

    head("Bravo").hover()
    check("pin group: offers to take the place", "in place of Charlie" in page.get_by_role("button", name="Pin group Bravo", exact=True).get_attribute("title"))
    page.get_by_role("button", name="Pin group Bravo", exact=True).click()
    page.wait_for_timeout(600)
    check("pin group: only one at a time", order(page)[:3] == ["Bravo", "Alpha", "Charlie"] and [g["name"] for g in server_groups(page) if g.get("pinned")] == ["Bravo"], str(order(page)))

    # ── Nothing above it ──
    check("pinned group can't be dragged", head("Bravo").get_attribute("draggable") is None)
    drag_to(page, head("Charlie").locator("button[aria-expanded]"), head("Bravo").locator("button[aria-expanded]"), y_frac=0.1)
    shot(page, "41-nothing-above-the-pin")
    page.mouse.up()
    page.wait_for_timeout(600)
    check("drop above the pin lands below it", order(page)[:3] == ["Bravo", "Charlie", "Alpha"], str(order(page)))

    page.reload()
    page.wait_for_selector('section[data-section="Bravo"]')
    close_guide(page)
    check("pin group: survives a reload", order(page)[0] == "Bravo" and pressed(page, "Unpin group Bravo"))

    # ── Unpinned: back in its place ──
    page.get_by_role("button", name="Unpin group Bravo", exact=True).click()
    page.wait_for_timeout(600)
    check("unpin group: back in its place", order(page)[:3] == ["Charlie", "Alpha", "Bravo"] and not [g for g in server_groups(page) if g.get("pinned")], str(order(page)))

    # ── Many agents in a group ──
    alpha = names_in(page, "Alpha")
    check("agents: running ones first to begin with", alpha[-1] == "Report Writer", str(alpha))
    row(page, "Report Writer").hover()
    shot(page, "42-row-pin-on-hover")
    page.get_by_role("button", name="Pin Report Writer", exact=True).click()
    page.wait_for_timeout(150)
    check("pin agent: goes first", names_in(page, "Alpha")[0] == "Report Writer", str(names_in(page, "Alpha")))
    row(page, "Research Assistant").hover()
    page.get_by_role("button", name="Pin Research Assistant", exact=True).click()
    page.wait_for_timeout(150)
    check("pin agent: many, in the order pinned", names_in(page, "Alpha")[:2] == ["Report Writer", "Research Assistant"], str(names_in(page, "Alpha")))
    row(page, "Grant Scout").hover()
    page.get_by_role("button", name="Pin Grant Scout", exact=True).click()
    page.mouse.move(800, 400)
    page.wait_for_timeout(600)
    shot(page, "43-agents-pinned")
    check("pin agent: each group its own", names_in(page, "Bravo")[0] == "Grant Scout", str(names_in(page, "Bravo")))
    stored = {g["name"]: g for g in server_groups(page)}
    check("pin agent: saved", stored["Alpha"]["pinnedAgentIds"] == ["proc-report-writer", "proc-research-assistant"] and stored["Bravo"]["pinnedAgentIds"] == ["proc-grant-scout"], json.dumps(stored["Alpha"])[:200])
    check("pin agent: stays shown without a hover", page.get_by_role("button", name="Unpin Report Writer", exact=True).evaluate("e => getComputedStyle(e).opacity") == "1")
    check("Other agents has no pins", page.get_by_role("button", name="Pin Home Admin", exact=True).count() == 0)

    page.reload()
    page.wait_for_selector('section[data-section="Alpha"]')
    close_guide(page)
    check("pin agent: survives a reload", names_in(page, "Alpha")[:2] == ["Report Writer", "Research Assistant"])

    # ── From the agent's menu ──
    row(page, "Market Watch").hover()
    page.get_by_role("button", name="More for Market Watch").click()
    menu = page.get_by_role("dialog", name="Market Watch")
    page.wait_for_timeout(350)
    shot(page, "44-menu-pin")
    menu.get_by_role("button", name="To the top of Alpha").click()
    page.wait_for_timeout(150)
    check("menu: pins", names_in(page, "Alpha")[2] == "Market Watch" and pressed(page, "Unpin Market Watch"), str(names_in(page, "Alpha")))
    row(page, "Market Watch").hover()
    page.get_by_role("button", name="More for Market Watch").click()
    page.get_by_role("dialog", name="Market Watch").get_by_role("button", name="Back among the others").click()
    page.wait_for_timeout(150)
    check("menu: unpins", pressed(page, "Pin Market Watch") is False and page.get_by_role("button", name="Unpin Market Watch", exact=True).count() == 0)

    # ── Leaving a group drops its pin ──
    page.get_by_role("button", name="Unpin Research Assistant", exact=True).click()
    page.wait_for_timeout(150)
    check("unpin agent: back among the others", names_in(page, "Alpha")[0] == "Report Writer" and page.get_by_role("button", name="Unpin Research Assistant", exact=True).count() == 0, str(names_in(page, "Alpha")))
    drag_to(page, row(page, "Report Writer"), head("Bravo").locator("button[aria-expanded]"))
    page.mouse.up()
    page.mouse.move(800, 400)
    page.wait_for_timeout(600)
    stored = {g["name"]: g for g in server_groups(page)}
    check("moved to another group: pin stays behind", "Report Writer" in names_in(page, "Bravo") and stored["Alpha"].get("pinnedAgentIds") == [] and "proc-report-writer" not in stored["Bravo"].get("pinnedAgentIds", []), json.dumps(stored)[:260])

    # ── The group's sheet ──
    head("Alpha").hover()
    page.get_by_role("button", name="Edit group Alpha").click()
    dialog = page.get_by_role("dialog", name="Edit group Alpha")
    page.wait_for_timeout(350)
    switch = dialog.get_by_role("switch", name="Pin to the top")
    check("sheet: switch off to begin with", switch.get_attribute("aria-checked") == "false")
    switch.click()
    page.wait_for_timeout(200)
    shot(page, "45-sheet-pinned")
    check("sheet: pins at once", switch.get_attribute("aria-checked") == "true" and order(page)[0] == "Alpha", str(order(page)))
    check("sheet: no order arrows while pinned", dialog.get_by_role("button", name="Higher in the list").count() == 0)
    dialog.get_by_role("button", name="Cancel").click()

    head("Bravo").hover()
    page.get_by_role("button", name="Edit group Bravo").click()
    dialog = page.get_by_role("dialog", name="Edit group Bravo")
    page.wait_for_timeout(350)
    check("sheet: says whose place it takes", dialog.get_by_text("Takes the place of Alpha").count() == 1)
    shot(page, "46-sheet-takes-the-place")
    dialog.get_by_role("checkbox", name="Grant Scout").click()
    dialog.get_by_role("button", name="Save").click()
    page.wait_for_timeout(600)
    stored = {g["name"]: g for g in server_groups(page)}
    check("sheet: an untick drops the pin", "proc-grant-scout" not in stored["Bravo"].get("pinnedAgentIds", []), json.dumps(stored["Bravo"])[:200])

    # ── A new group, pinned from the start ──
    page.get_by_title("New group").click()
    dialog = page.get_by_role("dialog", name="New group")
    page.wait_for_timeout(350)
    dialog.get_by_label("Group name").fill("Delta")
    dialog.get_by_role("switch", name="Pin to the top").click()
    check("new group: nothing pinned until it's created", order(page)[0] == "Alpha")
    dialog.get_by_role("button", name="Create group").click()
    page.wait_for_timeout(600)
    check("new group: pinned", order(page)[0] == "Delta" and [g["name"] for g in server_groups(page) if g.get("pinned")] == ["Delta"], str(order(page)))

    # ── Deleted: the next one doesn't inherit it ──
    head("Delta").hover()
    page.get_by_role("button", name="Edit group Delta").click()
    page.get_by_role("dialog", name="Edit group Delta").get_by_role("button", name="Delete").click()
    page.wait_for_timeout(600)
    check("delete the pinned group: nothing else pinned", not [g for g in server_groups(page) if g.get("pinned")])

    # ── Two devices each pinned one ──
    put_groups(page, [
        {"id": "g-a", "name": "Alpha", "color": "violet", "type": "group", "agentIds": ["proc-research-assistant"], "pinned": True},
        {"id": "g-b", "name": "Bravo", "color": "blue", "type": "group", "agentIds": ["proc-inbox-helper"], "pinned": True, "pinnedAgentIds": ["proc-inbox-helper", "proc-not-in-it"]},
    ])
    page.reload()
    page.wait_for_selector('section[data-section="Alpha"]')
    close_guide(page)
    check("two pinned in the saved data: the first holds it", page.get_by_role("button", name="Unpin group Alpha", exact=True).count() == 1 and page.get_by_role("button", name="Unpin group Bravo", exact=True).count() == 0)

    # ── A pinned agent on the rail ──
    put_groups(page, [
        {"id": "g-a", "name": "Alpha", "color": "violet", "type": "group", "agentIds": ["proc-research-assistant", "proc-report-writer"], "pinnedAgentIds": ["proc-report-writer"]},
        {"id": "g-b", "name": "Bravo", "color": "blue", "type": "group", "agentIds": ["proc-inbox-helper", "proc-grant-scout"], "pinned": True, "pinnedAgentIds": ["proc-grant-scout"]},
    ])
    page.reload()
    page.wait_for_selector('section[data-section="Alpha"]')
    close_guide(page)
    page.get_by_title("Hide agents").click()
    page.wait_for_timeout(250)
    shot(page, "47-rail-pins")
    titles = page.locator("aside button[aria-disabled], aside button[title^='Chat with']").evaluate_all("els => els.map(e => e.title)")
    check("rail: follows the pins", "Grant Scout" in titles[0] and "Report Writer" in titles[2], str(titles))
    page.get_by_title("Show agents").click()
    page.wait_for_timeout(200)

    # ── Light ──
    page.get_by_role("button", name="Settings").click()
    page.wait_for_timeout(300)
    page.get_by_role("switch", name="Dark theme").click()
    page.keyboard.press("Escape")
    page.mouse.move(800, 400)
    page.wait_for_timeout(400)
    shot(page, "48-light-pins")
    page.get_by_role("button", name="Settings").click()
    page.wait_for_timeout(300)
    page.get_by_role("switch", name="Dark theme").click()
    page.keyboard.press("Escape")
    page.wait_for_timeout(500)
    check("pins: no page errors", not errors, "; ".join(errors)[:300])
    ctx.close()

    # ── A phone: the menu and the sheet ──
    ctx = browser.new_context(viewport={"width": 390, "height": 844}, has_touch=True, is_mobile=True)
    ctx.add_init_script("try{localStorage.setItem('fd:theme','dark')}catch{}")
    page = ctx.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    sign_in(page, "maya@example.com")
    page.wait_for_selector('section[data-section="Alpha"]')
    close_guide(page)
    check("phone: order from the server", order(page)[:2] == ["Bravo", "Alpha"], str(order(page)))
    check("phone: the pinned group is marked", section(page, "Bravo").locator(".group\\/head svg.lucide-pin").count() == 1)
    check("phone: a pinned agent is marked", section(page, "Bravo").locator("li", has_text="Grant Scout").first.locator("svg.lucide-pin").count() == 1)
    shot(page, "49-phone-pins")
    page.get_by_role("button", name="More for Research Assistant").tap()
    menu = page.get_by_role("dialog", name="Research Assistant")
    page.wait_for_timeout(350)
    menu.get_by_role("button", name="To the top of Alpha").scroll_into_view_if_needed()
    menu.get_by_role("button", name="To the top of Alpha").tap()
    page.wait_for_timeout(600)
    stored = {g["name"]: g for g in server_groups(page)}
    check("phone: pins an agent from its menu", stored["Alpha"]["pinnedAgentIds"] == ["proc-report-writer", "proc-research-assistant"], json.dumps(stored["Alpha"])[:200])
    page.get_by_role("button", name="Edit group Alpha").tap()
    dialog = page.get_by_role("dialog", name="Edit group Alpha")
    page.wait_for_timeout(350)
    shot(page, "50-phone-sheet")
    dialog.get_by_role("switch", name="Pin to the top").tap()
    page.wait_for_timeout(150)
    dialog.get_by_role("button", name="Cancel").tap()
    page.wait_for_timeout(600)
    check("phone: pins a group from its sheet", order(page)[0] == "Alpha" and [g["name"] for g in server_groups(page) if g.get("pinned")] == ["Alpha"], str(order(page)))
    check("phone: pins, no page errors", not errors, "; ".join(errors)[:300])
    ctx.close()


def admin(browser):
    # The desktop app's "user view": the same locked kiosk a user gets.
    ctx = browser.new_context(viewport={"width": 1280, "height": 860})
    ctx.add_init_script("try{localStorage.setItem('fd:theme','dark')}catch{}")
    page = ctx.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(BASE + "/?app=desktop")
    page.fill("input[type=email]", "admin@example.com")
    page.fill("input[type=password]", PASSWORD)
    page.click("button[type=submit]")
    page.wait_for_timeout(1500)
    if page.get_by_text("Switch to user view").count():
        page.get_by_text("Switch to user view").first.click()
    page.wait_for_selector("aside li")
    close_guide(page)
    check("admin, desktop app: says User view", page.get_by_text("User view").count() >= 1)
    check("admin: only their own agents", page.locator("aside li").count() == 3, str(page.locator("aside li").count()))
    check("admin: Maya's groups aren't theirs", page.locator("section[data-section]").count() == 0, str(order(page)))
    page.get_by_title("New group").click()
    dialog = page.get_by_role("dialog", name="New group")
    dialog.get_by_label("Group name").fill("Ops")
    dialog.get_by_role("checkbox", name="Admin Scout").click()
    dialog.get_by_role("button", name="Create group").click()
    expect(section(page, "Ops")).to_be_visible()
    drag_to(page, row(page, "Admin Clerk"), section(page, "Ops").locator(HEAD))
    page.mouse.up()
    page.wait_for_timeout(250)
    check("admin, desktop app: drag works", "Admin Clerk" in names_in(page, "Ops"), str(names_in(page, "Ops")))
    row(page, "Admin Third").hover()
    page.get_by_role("button", name="More for Admin Third").click()
    menu = page.get_by_role("dialog", name="Admin Third")
    check("admin, desktop app: the kiosk's full menu", menu.get_by_role("button", name="Settings").count() == 1 and menu.get_by_role("button", name="Group").count() == 1)
    menu.get_by_role("button", name="Group").click()
    picker = page.get_by_role("dialog", name="Group for Admin Third")
    picker.get_by_role("radio", name="Ops").click()
    check("admin, desktop app: picker works", "Admin Third" in names_in(page, "Ops"))
    page.wait_for_timeout(500)
    shot(page, "31-admin-user-view")
    check("admin, desktop app: no page errors", not errors, "; ".join(errors)[:300])
    ctx.close()

    # On the web: Open Flight Deck, then the simple layout (nothing locked).
    ctx = browser.new_context(viewport={"width": 1280, "height": 860})
    ctx.add_init_script("try{localStorage.setItem('fd:theme','dark')}catch{}")
    page = ctx.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    sign_in(page, "admin@example.com")
    page.get_by_role("button", name="Open Flight Deck").first.click()
    page.wait_for_timeout(1200)
    page.keyboard.press("Control+\\")
    page.wait_for_timeout(300)
    if page.locator("section[data-section]").count() == 0:
        page.locator("button:has(svg.lucide-columns-3)").first.click()
    page.wait_for_selector('section[data-section="Ops"]')
    shot(page, "32-admin-web-simple")
    check("admin, web: their groups, from the server", sorted(names_in(page, "Ops")) == ["Admin Clerk", "Admin Scout", "Admin Third"], str(names_in(page, "Ops")))
    row(page, "Admin Third").hover()
    page.get_by_role("button", name="More for Admin Third").click()
    menu = page.get_by_role("dialog", name="Admin Third")
    check("admin, web: \u22ef holds the agent's group", menu.get_by_role("button", name="Group").count() == 1)
    menu.get_by_role("button", name="Group").click()
    picker = page.get_by_role("dialog", name="Group for Admin Third")
    picker.get_by_role("radio", name="No group").click()
    page.wait_for_timeout(250)
    check("admin, web: taken out", "Admin Third" in names_in(page, "Other agents"), str(order(page)))
    check("admin, web: no page errors", not errors, "; ".join(errors)[:300])
    ctx.close()


def no_sign_in(browser):
    """A deck nobody signs in to (the desktop app): groups live in this browser. The deck's answers are stubbed."""
    ctx = browser.new_context(viewport={"width": 1280, "height": 860})
    ctx.add_init_script("try{localStorage.setItem('fd:theme','dark')}catch{}")
    page = ctx.new_page()
    puts = []
    agents = [{"slug": n, "name": n.title(), "description": "", "status": "running", "web_port": 1, "web_auth": "", "provider": "", "model": ""} for n in ("alpha", "bravo", "charlie")]

    def on_route(route):
        url = route.request.url
        if "/fd/auth/status" in url:
            return route.fulfill(json={"auth_enabled": False, "simple_chat_only": True})
        if "/fd/processes" in url:
            return route.fulfill(json=agents)
        if "/fd/lesson-proposals" in url:  # an object: the kiosk "⋯" reads its .agents
            return route.fulfill(json={"agents": []})
        if "/fd/settings" in url and route.request.method != "GET":
            puts.append(url)
            return route.fulfill(json={})
        if "/fd/" in url and route.request.method == "GET":
            return route.fulfill(json=[])
        if "/fd/" in url:
            return route.fulfill(json={})
        return route.continue_()

    page.route("**/*", on_route)
    page.goto(BASE)
    page.wait_for_selector("aside li")
    close_guide(page)
    page.get_by_title("New group").click()
    dialog = page.get_by_role("dialog", name="New group")
    dialog.get_by_label("Group name").fill("Team")
    dialog.get_by_role("checkbox", name="Bravo").click()
    dialog.get_by_role("button", name="Create group").click()
    expect(section(page, "Team")).to_be_visible()
    page.reload()
    page.wait_for_selector('section[data-section="Team"]')
    close_guide(page)
    check("no sign-in: groups survive a reload", names_in(page, "Team") == ["Bravo"], str(names_in(page, "Team")))
    check("no sign-in: nothing sent to the server", not puts, str(puts))
    ctx.close()


with sync_playwright() as p:
    browser = p.chromium.launch()
    which = sys.argv[1:] or ["kiosk", "phone", "pins", "admin", "no-sign-in"]
    try:
        if "kiosk" in which:
            kiosk_user(browser)
        if "phone" in which:
            phone(browser)
        if "pins" in which:
            pins(browser)
        if "admin" in which:
            admin(browser)
        if "no-sign-in" in which:
            no_sign_in(browser)
    finally:
        browser.close()

failed = [r for r in results if not r[1]]
print(f"\n{len(results) - len(failed)}/{len(results)} passed")
for name, _, detail in failed:
    print("FAILED:", name, detail)
sys.exit(1 if failed else 0)
