"""One shared definition of what a page *shows* to a text agent, in document order.

Both drivers must produce the same item sequence for the same HTML (a parity test enforces it):
the in-process driver parses HTML with `parse_items`; the Playwright driver runs `COLLECT_JS`,
a DOM walk implementing the same rules:

  - text: visible text outside interactive/label/excluded elements, merged per block and
    whitespace-normalised; one item per block;
  - input: name/type/autocomplete/id and its label (text of the enclosing <label>);
  - button: its own text, optional `formaction`;
  - a: its own text and href;
  - iframe: a placeholder at its position (the driver inlines the child document there);
  - form membership: every input/button carries its form's key, action and method.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

BLOCK_TAGS = {
    "p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "ul", "ol", "section", "header",
    "footer", "main", "article", "form", "table", "tr", "td", "th", "br", "hr", "body", "nav",
    "aside", "blockquote", "pre", "dl", "dt", "dd",
}
EXCLUDED_TAGS = {"script", "style", "title", "head", "option", "select", "textarea", "svg", "noscript", "template"}
_VOID = {"input", "br", "hr", "img", "meta", "link", "area", "base", "col", "embed", "source", "track", "wbr"}
_WS = re.compile(r"\s+")


def norm_text(s: str) -> str:
    return _WS.sub(" ", s or "").strip()


class _ItemParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.items: list[dict] = []
        self.title = ""
        self._stack: list[str] = []
        self._excluded = 0
        self._text = ""
        self._in_title = False
        self._label: list[str] | None = None  # active <label> text buffer
        self._label_used = False
        self._link: dict | None = None
        self._button: dict | None = None
        self._form: dict | None = None
        self._nform = 0

    # ---- helpers ----
    def _flush(self) -> None:
        t = norm_text(self._text)
        if t:
            self.items.append({"kind": "text", "text": t})
        self._text = ""

    def _form_fields(self) -> dict:
        if self._form is None:
            return {"form_key": "", "form_action": None, "form_method": ""}
        return {"form_key": self._form["key"], "form_action": self._form["action"],
                "form_method": self._form["method"]}

    # ---- parser callbacks ----
    def handle_starttag(self, tag, attrs):
        a = {k: (v if v is not None else "") for k, v in attrs}
        if a.get("aria-hidden", "").lower() == "true" and tag not in _VOID:
            # hidden from assistive tech (and from text agents); its pixels may still render
            self._excluded += 1
            self._stack.append("\x00hidden:" + tag)
            return
        if tag in EXCLUDED_TAGS:
            if tag == "title":
                self._in_title = True
            if tag not in _VOID:
                self._excluded += 1
                self._stack.append(tag)
            return
        if self._excluded:
            if tag not in _VOID:
                self._stack.append(tag)
            return
        if tag in BLOCK_TAGS:
            self._flush()
        if tag == "form":
            self._nform += 1
            self._form = {"key": f"form{self._nform}", "action": a.get("action", ""),
                          "method": (a.get("method") or "get").lower()}
        elif tag == "label":
            self._flush()
            self._label = []
            self._label_used = False
        elif tag == "input":
            itype = (a.get("type") or "text").lower()
            if itype not in ("hidden", "submit", "button", "image", "reset"):
                label = norm_text("".join(self._label)) if self._label is not None else ""
                if self._label is not None:
                    self._label_used = True
                self._flush()
                self.items.append({
                    "kind": "input", "name": a.get("name", ""), "type": itype,
                    "autocomplete": a.get("autocomplete", ""), "id": a.get("id", ""),
                    "label": label, **self._form_fields(),
                })
            elif itype == "submit":
                self._flush()
                self.items.append({"kind": "button", "label": norm_text(a.get("value", "") or "Submit"),
                                   "formaction": a.get("formaction"), **self._form_fields()})
        elif tag == "button":
            self._flush()
            self._button = {"kind": "button", "label": "", "formaction": a.get("formaction"),
                            "btype": (a.get("type") or "submit").lower(), **self._form_fields()}
        elif tag == "a":
            self._flush()
            self._link = {"kind": "a", "label": "", "href": a.get("href", ""), "id": a.get("id", "")}
        elif tag == "iframe":
            self._flush()
            self.items.append({"kind": "iframe", "src": a.get("src", "")})
        if tag not in _VOID:
            self._stack.append(tag)

    def handle_endtag(self, tag):
        # pop to the matching tag (tolerate unclosed elements)
        if tag in self._stack or ("\x00hidden:" + tag) in self._stack:
            while self._stack:
                t = self._stack.pop()
                hidden = t.startswith("\x00hidden:")
                base = t.split(":", 1)[1] if hidden else t
                if (hidden or t in EXCLUDED_TAGS) and self._excluded:
                    self._excluded -= 1
                if t == "title":
                    self._in_title = False
                if base == tag:
                    break
        if self._excluded:
            return
        if tag == "a" and self._link is not None:
            self._link["label"] = norm_text(self._link["label"])
            self.items.append(self._link)
            self._link = None
        elif tag == "button" and self._button is not None:
            self._button["label"] = norm_text(self._button["label"])
            self.items.append(self._button)
            self._button = None
        elif tag == "label":
            if self._label is not None and not self._label_used:
                # a label with no nested input: keep its text as visible text
                t = norm_text("".join(self._label))
                if t:
                    self.items.append({"kind": "text", "text": t})
            self._label = None
        elif tag == "form":
            self._flush()
            self._form = None
        elif tag in BLOCK_TAGS:
            self._flush()

    def handle_data(self, data):
        if self._in_title:
            self.title += data
            return
        if self._excluded:
            return
        if self._link is not None:
            self._link["label"] += data
        elif self._button is not None:
            self._button["label"] += data
        elif self._label is not None and not self._label_used:
            self._label.append(data)
        elif self._label is not None:
            pass  # text after the input inside a label (hints) is not shown separately
        else:
            self._text += data

    def close(self):
        super().close()
        self._flush()


def parse_items(html: str) -> tuple[str, list[dict]]:
    """Return (title, items) for one HTML document."""
    p = _ItemParser()
    p.feed(html or "")
    p.close()
    return norm_text(p.title), p.items


# The same rules as a DOM walk, for Chromium. `prefix` makes per-frame element tokens unique.
COLLECT_JS = r"""
(prefix) => {
  const BLOCK = new Set(%BLOCK%);
  const EXCL = new Set(%EXCL%);
  const norm = s => (s || '').replace(/\s+/g, ' ').trim();
  const items = []; let n = 0; let text = '';
  const flush = () => { const t = norm(text); if (t) items.push({kind: 'text', text: t}); text = ''; };
  const formAction = (f) => {
    try { return Object.getOwnPropertyDescriptor(HTMLFormElement.prototype, 'action').get.call(f); }
    catch (e) { return ''; }
  };
  const formMethod = (f) => {
    try { return (Object.getOwnPropertyDescriptor(HTMLFormElement.prototype, 'method').get.call(f) || 'get').toLowerCase(); }
    catch (e) { return 'get'; }
  };
  const forms = new Map(); let nform = 0;
  const formFields = (el) => {
    const f = el.form;
    if (!f) return {form_key: '', form_action: null, form_method: ''};
    if (!forms.has(f)) { nform += 1; forms.set(f, 'form' + nform); }
    return {form_key: forms.get(f), form_action: formAction(f), form_method: formMethod(f)};
  };
  const tag = (el) => { const tok = prefix + '_' + (n++); el.setAttribute('data-ob-id', tok); return tok; };
  const labelText = (lab, input) => {
    let s = '';
    const walk = (node) => {
      for (const c of node.childNodes) {
        if (c === input) return true;
        if (c.nodeType === 3) s += c.nodeValue;
        else if (c.nodeType === 1 && !EXCL.has(c.tagName.toLowerCase())) { if (walk(c)) return true; }
      }
      return false;
    };
    walk(lab);
    return norm(s);
  };
  const visit = (node, inLabel) => {
    for (const c of node.childNodes) {
      if (c.nodeType === 3) { if (!inLabel) text += c.nodeValue; continue; }
      if (c.nodeType !== 1) continue;
      const t = c.tagName.toLowerCase();
      if (EXCL.has(t)) continue;
      if ((c.getAttribute('aria-hidden') || '').toLowerCase() === 'true') continue;
      if (t === 'input') {
        const it = (c.getAttribute('type') || 'text').toLowerCase();
        if (['hidden', 'button', 'image', 'reset'].includes(it)) continue;
        flush();
        if (it === 'submit') {
          items.push(Object.assign({kind: 'button', label: norm(c.value || 'Submit'), formaction: c.getAttribute('formaction'), tok: tag(c)}, formFields(c)));
          continue;
        }
        const lab = c.closest('label');
        items.push(Object.assign({kind: 'input', name: c.getAttribute('name') || '', type: it,
          autocomplete: c.getAttribute('autocomplete') || '', id: c.id || '',
          label: lab ? labelText(lab, c) : '', tok: tag(c)}, formFields(c)));
        continue;
      }
      if (t === 'button') {
        flush();
        items.push(Object.assign({kind: 'button', label: norm(c.textContent), formaction: c.getAttribute('formaction'),
          btype: (c.getAttribute('type') || 'submit').toLowerCase(), tok: tag(c)}, formFields(c)));
        continue;
      }
      if (t === 'a') {
        flush();
        items.push({kind: 'a', label: norm(c.textContent), href: c.getAttribute('href') ? c.href : '', id: c.id || '', tok: tag(c)});
        continue;
      }
      if (t === 'iframe') { flush(); items.push({kind: 'iframe', src: c.getAttribute('src') ? c.src : '', tok: tag(c)}); continue; }
      if (t === 'label') {
        flush();
        const hasInput = !!c.querySelector('input');
        if (!hasInput) { const lt = norm(c.textContent); if (lt) items.push({kind: 'text', text: lt}); continue; }
        visit(c, true);
        continue;
      }
      const block = BLOCK.has(t);
      if (block) flush();
      visit(c, inLabel);
      if (block) flush();
    }
  };
  visit(document.body || document.documentElement, false);
  flush();
  return {title: norm(document.title), items};
}
""".replace("%BLOCK%", str(sorted(BLOCK_TAGS))).replace("%EXCL%", str(sorted(EXCLUDED_TAGS)))
