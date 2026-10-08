/* Shared notifications.
 *
 * showToast(msg, ok|'warn', {duration, kind})
 *   Stacked, wrapping toasts. The display time grows with the message length
 *   (warnings and errors stay longer), pauses while the pointer is over the
 *   toast, and the toast can be dismissed with × or a click on it.
 *
 * showNotice(title, items, {kind})
 *   Persistent dialog listing ALL messages (warnings, import errors, ...).
 *   It stays until closed (button, Escape, click outside) and offers "Copy".
 */
(function () {
  function stack() {
    let s = document.getElementById('toast-stack');
    if (!s) {
      s = document.createElement('div');
      s.id = 'toast-stack';
      s.className = 'toast-stack';
      document.body.appendChild(s);
    }
    return s;
  }

  window.showToast = function (msg, ok, opts) {
    opts = opts || {};
    msg = String(msg);
    const kind = opts.kind || (ok === true ? 'ok' : ok === 'warn' ? 'warn' : 'err');
    const el = document.createElement('div');
    el.className = 'toast toast-' + kind;
    el.setAttribute('role', kind === 'ok' ? 'status' : 'alert');

    const text = document.createElement('span');
    text.className = 'toast-text';
    text.textContent = msg;
    const x = document.createElement('button');
    x.type = 'button';
    x.className = 'toast-x';
    x.setAttribute('aria-label', 'Dismiss');
    x.textContent = '\u00d7';
    el.append(text, x);
    stack().appendChild(el);

    // ~70 ms per character, at least 4 s, at most 30 s; 1.6x for warnings/errors
    let remaining = opts.duration != null
      ? opts.duration
      : Math.min(30000, Math.max(4000, msg.length * 70)) * (kind === 'ok' ? 1 : 1.6);
    let timer = null, started = 0;
    const close = () => { clearTimeout(timer); el.style.opacity = '0'; setTimeout(() => el.remove(), 300); };
    const start = () => { started = Date.now(); timer = setTimeout(close, remaining); };
    el.addEventListener('mouseenter', () => { clearTimeout(timer); remaining -= Date.now() - started; });
    el.addEventListener('mouseleave', () => { remaining = Math.max(remaining, 2500); start(); });
    el.addEventListener('click', close);
    start();
    return el;
  };

  window.showNotice = function (title, items, opts) {
    items = (items || []).map(String).filter(Boolean);
    if (!items.length) return;
    opts = opts || {};

    const overlay = document.createElement('div');
    overlay.className = 'notice-overlay';
    const box = document.createElement('div');
    box.className = 'notice notice-' + (opts.kind || 'warn');
    box.setAttribute('role', 'dialog');
    box.setAttribute('aria-modal', 'true');

    const h = document.createElement('h3');
    h.textContent = title + ' (' + items.length + ')';
    const ul = document.createElement('ul');
    items.forEach(t => { const li = document.createElement('li'); li.textContent = t; ul.appendChild(li); });

    const actions = document.createElement('div');
    actions.className = 'notice-actions';
    const copy = document.createElement('button');
    copy.type = 'button';
    copy.className = 'btn btn-ghost';
    copy.textContent = 'Copy all';
    const closeBtn = document.createElement('button');
    closeBtn.type = 'button';
    closeBtn.className = 'btn btn-primary';
    closeBtn.textContent = 'Close';
    actions.append(copy, closeBtn);
    box.append(h, ul, actions);
    overlay.appendChild(box);
    document.body.appendChild(overlay);

    const close = () => { document.removeEventListener('keydown', onKey); overlay.remove(); };
    const onKey = e => { if (e.key === 'Escape') close(); };
    document.addEventListener('keydown', onKey);
    overlay.addEventListener('click', e => { if (e.target === overlay) close(); });
    closeBtn.addEventListener('click', close);
    copy.addEventListener('click', () => {
      const all = items.join('\n');
      const done = () => { copy.textContent = 'Copied'; };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(all).then(done, () => {});
      } else {
        const ta = document.createElement('textarea');
        ta.value = all; document.body.appendChild(ta); ta.select();
        try { document.execCommand('copy'); done(); } catch (_) {}
        ta.remove();
      }
    });
    closeBtn.focus();
  };
})();
