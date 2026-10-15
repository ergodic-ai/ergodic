/* API-122 / DEC-073: local-only documentation interactions. No telemetry, credentials or live execution.
   Vendored from welt-python docs/assets/docs.js (see docsite/provenance.json). API-143: the theme storage key and
   the search prompts come from data attributes the builder sets from docsite/site.toml. */
(() => {
  const root = document.documentElement;
  const announce = (value) => { const live = document.querySelector('#announcement'); live.textContent = ''; requestAnimationFrame(() => { live.textContent = value; }); };

  // Theme follows the system until the reader chooses; only that display choice is stored.
  const theme = document.querySelector('#theme');
  const dark = () => (root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches);
  const labelTheme = () => theme.setAttribute('aria-label', dark() ? 'Switch to light theme' : 'Switch to dark theme');
  labelTheme();
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', labelTheme);
  theme.addEventListener('click', () => { root.dataset.theme = dark() ? 'light' : 'dark'; try { localStorage.setItem(root.dataset.themeKey || 'docs-theme', root.dataset.theme); } catch (_) {} labelTheme(); announce(`${root.dataset.theme === 'dark' ? 'Dark' : 'Light'} theme.`); });

  // Copy buttons; a complete program in data-copy wins over the visible code.
  const copyText = async (text, button, label) => {
    await navigator.clipboard.writeText(text);
    button.textContent = 'Copied'; announce('Copied to clipboard.');
    setTimeout(() => { button.textContent = label; }, 2000);
  };
  document.querySelectorAll('.code-wrap').forEach((wrap) => {
    const label = wrap.dataset.copyLabel || 'Copy';
    const button = document.createElement('button'); button.className = 'copy'; button.type = 'button'; button.textContent = label; button.setAttribute('aria-label', wrap.dataset.copyLabel || 'Copy code'); wrap.append(button);
    button.addEventListener('click', async () => {
      const code = [...wrap.querySelectorAll('pre')].find((pre) => !pre.closest('[hidden]')).querySelector('code');
      try { await copyText(wrap.dataset.copy || code.textContent, button, label); } catch (_) {
        button.textContent = 'Select code';
        if (wrap.dataset.copy) { let fallback = wrap.querySelector('.copy-fallback'); if (!fallback) { fallback = document.createElement('textarea'); fallback.className = 'copy-fallback'; fallback.readOnly = true; fallback.rows = 12; fallback.setAttribute('aria-label', 'Complete Python example to select and copy'); fallback.value = wrap.dataset.copy; wrap.append(fallback); } fallback.focus(); fallback.select(); announce('Clipboard unavailable. The complete example is selected below; copy it with your keyboard.'); }
        else { const selection = window.getSelection(); const range = document.createRange(); range.selectNodeContents(code); selection.removeAllRanges(); selection.addRange(range); announce('Clipboard unavailable. The code is selected; copy it with your keyboard.'); }
      }
    });
  });
  document.querySelectorAll('[data-markdown]').forEach((button) => {
    const label = button.querySelector('span'); const text = label.textContent;
    button.addEventListener('click', async () => {
      try { const response = await fetch(button.dataset.markdown); if (!response.ok) throw Error(); await copyText(await response.text(), label, text); }
      catch (_) { location.href = button.dataset.markdown; }
    });
  });

  // Tabs with roving focus.
  document.querySelectorAll('[role=tablist]').forEach((list) => {
    const tabs = [...list.querySelectorAll('[role=tab]')];
    const select = (tab) => { tabs.forEach((item) => { const active = item === tab; item.setAttribute('aria-selected', String(active)); item.tabIndex = active ? 0 : -1; document.getElementById(item.getAttribute('aria-controls')).hidden = !active; }); };
    tabs.forEach((tab, i) => { tab.addEventListener('click', () => select(tab)); tab.addEventListener('keydown', (event) => { let next; if (event.key === 'ArrowRight') next = tabs[(i + 1) % tabs.length]; if (event.key === 'ArrowLeft') next = tabs[(i + tabs.length - 1) % tabs.length]; if (event.key === 'Home') next = tabs[0]; if (event.key === 'End') next = tabs[tabs.length - 1]; if (next) { event.preventDefault(); select(next); next.focus(); } }); });
  });

  document.querySelectorAll('[data-version]').forEach((select) => select.addEventListener('change', (event) => { location.href = event.target.value; }));
  if (!/Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent)) document.querySelectorAll('.shortcut').forEach((hint) => { hint.textContent = 'Ctrl K'; });

  // Outline: mark the section currently under the top bar.
  const links = [...document.querySelectorAll('.outline-inner a[href^="#"]')];
  const targets = links.map((link) => document.getElementById(decodeURIComponent(link.hash.slice(1)))).filter(Boolean);
  if (targets.length) {
    const spy = () => { let current = targets[0]; targets.forEach((target) => { if (target.getBoundingClientRect().top < 120) current = target; }); links.forEach((link) => { const active = decodeURIComponent(link.hash.slice(1)) === current.id; link.classList.toggle('is-active', active); if (active) link.setAttribute('aria-current', 'location'); else link.removeAttribute('aria-current'); }); };
    addEventListener('scroll', spy, { passive: true }); spy();
  }

  // Dialogs return focus to their opener.
  const openDialog = (id, opener) => { const dialog = document.getElementById(id); dialog.showModal(); const restore = () => { dialog.removeEventListener('close', restore); if (dialog.dataset.keepFocus) delete dialog.dataset.keepFocus; else opener.focus(); }; dialog.addEventListener('close', restore); };
  document.querySelector('#menu').addEventListener('click', (event) => openDialog('navigation', event.currentTarget));
  document.querySelectorAll('dialog [data-close]').forEach((button) => button.addEventListener('click', () => button.closest('dialog').close()));
  document.querySelectorAll('dialog').forEach((dialog) => {
    dialog.addEventListener('click', (event) => { if (event.target === dialog) dialog.close(); });
    dialog.addEventListener('keydown', (event) => { if (event.key === 'Escape') { event.preventDefault(); dialog.close(); } });
  });

  // ⌘K search palette over this release's search index.
  const searchButton = document.querySelector('#search'); const search = document.querySelector('#search-dialog');
  const searchURL = new URL(document.body.dataset.search, location.href);
  let index; let selected = 0;
  const field = document.querySelector('#search-input'); const results = document.querySelector('#search-results'); const empty = document.querySelector('#search-empty');
  const searchRank = (item, query) => {
    const method = (item.method || '').toLowerCase();
    if (method && method === query) return 3;
    if (method && method.split('.').pop() === query) return 2;
    return item.title.toLowerCase().includes(query) ? 1 : 0;
  };
  const highlight = () => { [...results.children].forEach((li, i) => { li.setAttribute('aria-selected', String(i === selected)); if (i === selected) { field.setAttribute('aria-activedescendant', li.id); li.scrollIntoView({ block: 'nearest' }); } }); };
  const render = () => {
    results.replaceChildren(); selected = 0; field.removeAttribute('aria-activedescendant');
    const query = field.value.toLowerCase().trim(); const words = query.split(/\s+/).filter(Boolean);
    const found = words.length && index ? index.filter((item) => words.every((word) => `${item.title} ${item.text}`.toLowerCase().includes(word))).sort((a, b) => searchRank(b, query) - searchRank(a, query)).slice(0, 12) : [];
    empty.hidden = found.length > 0;
    empty.textContent = !words.length ? (document.body.dataset.searchPrompt || 'Search this version by task, function or error.') : !index ? 'Search is unavailable. Browse Guides or Reference in the navigation.' : (document.body.dataset.searchHint || 'No matches. Try a shorter query.');
    found.forEach((item, i) => { const li = document.createElement('li'); li.id = `search-result-${i}`; li.setAttribute('role', 'option'); const a = document.createElement('a'); a.href = new URL(item.href, searchURL).href; a.tabIndex = -1; a.textContent = item.title; const small = document.createElement('small'); small.textContent = item.text.slice(0, 140); a.append(small); li.append(a); results.append(li); });
    field.setAttribute('aria-expanded', String(found.length > 0));
    if (found.length) highlight();
    announce(found.length ? `${found.length} search results.` : empty.textContent);
  };
  const openSearch = async () => { if (search.open) return; const previous = document.activeElement; openDialog('search-dialog', previous && previous !== document.body && previous.isConnected ? previous : searchButton); field.focus(); field.select(); if (!index) { try { const response = await fetch(searchURL); if (!response.ok) throw Error(); index = await response.json(); } catch (_) { index = null; } } render(); };
  // A result on this page moves focus to its heading instead of back to the search button.
  const go = (link) => {
    const target = new URL(link.href);
    if (target.pathname !== location.pathname) { location.href = target.href; return; }
    search.dataset.keepFocus = 'true'; search.close(); history.pushState(null, '', target.hash || location.pathname);
    const heading = target.hash && document.getElementById(decodeURIComponent(target.hash.slice(1)));
    if (heading) { heading.tabIndex = -1; heading.scrollIntoView(); heading.focus({ preventScroll: true }); }
  };
  results.addEventListener('click', (event) => { const link = event.target.closest('a'); if (link) { event.preventDefault(); go(link); } });
  searchButton.addEventListener('click', openSearch);
  field.addEventListener('input', render);
  field.addEventListener('keydown', (event) => {
    const count = results.children.length;
    if (event.key === 'ArrowDown' && count) { event.preventDefault(); selected = (selected + 1) % count; highlight(); }
    if (event.key === 'ArrowUp' && count) { event.preventDefault(); selected = (selected + count - 1) % count; highlight(); }
    if (event.key === 'Enter' && count) { event.preventDefault(); go(results.children[selected].querySelector('a')); }
  });
  document.addEventListener('keydown', (event) => {
    const typing = ['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement.tagName) || document.activeElement.isContentEditable;
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') { event.preventDefault(); if (search.open) search.close(); else if (!document.querySelector('dialog[open]')) openSearch(); }
    else if (event.key === '/' && !typing && !document.querySelector('dialog[open]')) { event.preventDefault(); openSearch(); }
  });
})();
