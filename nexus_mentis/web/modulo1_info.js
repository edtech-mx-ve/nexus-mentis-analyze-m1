/* NEXUS ANALYZE M1 - Navegacion accesible, sin cuestionarios ni llamadas a la API. */
(() => {
  'use strict';
  const ids = Object.freeze(['acerca','dataset','fundamentos','trivia','ayuda','institucional']);
  const setup = () => {
    const tabs = ids.map(id => document.getElementById('tab-' + id));
    const panels = ids.map(id => document.getElementById('panel-' + id));
    if (tabs.some(el => !el) || panels.some(el => !el)) return;
    const activate = (index, focus) => {
      tabs.forEach((tab, position) => {
        const selected = position === index;
        tab.setAttribute('aria-selected', String(selected));
        tab.tabIndex = selected ? 0 : -1;
        panels[position].hidden = !selected;
      });
      if (focus) tabs[index].focus();
    };
    tabs.forEach((tab, position) => {
      tab.addEventListener('click', () => activate(position, false));
      tab.addEventListener('keydown', event => {
        let next;
        switch (event.key) {
          case 'ArrowRight': next = (position + 1) % tabs.length; break;
          case 'ArrowLeft': next = (position + tabs.length - 1) % tabs.length; break;
          case 'Home': next = 0; break;
          case 'End': next = tabs.length - 1; break;
          default: return;
        }
        event.preventDefault();
        activate(next, true);
      });
    });
    activate(0, false);
  };
  document.addEventListener('DOMContentLoaded', setup);
})();
