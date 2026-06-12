(function () {
  'use strict';
  var ARC = 188.5;   // length of the 270-degree knob arc

  function initKnob(el, p) {
    var val = el.querySelector('.val');
    var ptr = el.querySelector('.ptr');
    var out = el.querySelector('output');
    val.setAttribute('stroke-dasharray', ARC);

    function clamp(v) { return Math.min(p.max, Math.max(p.min, v)); }
    function paint(v) {
      var f = (v - p.min) / (p.max - p.min);
      val.setAttribute('stroke-dashoffset', ARC * (1 - f));
      ptr.setAttribute('transform', 'rotate(' + (-135 + 270 * f) + ' 50 50)');
      out.textContent = v.toFixed(1) + (p.unit ? ' ' + p.unit : '');
    }
    dspark.onParam(p.id, paint);   // host automation and our own edits

    var startY = 0, startV = 0, dragging = false;
    el.addEventListener('pointerdown', function (e) {
      dragging = true; startY = e.clientY; startV = dspark.getParam(p.id);
      el.setPointerCapture(e.pointerId);
      dspark.beginEdit(p.id);
      e.preventDefault();
    });
    el.addEventListener('pointermove', function (e) {
      if (!dragging) { return; }
      var pixelsForFullRange = e.shiftKey ? 1600 : 190;
      dspark.setParam(p.id, clamp(
        startV + (startY - e.clientY) * (p.max - p.min) / pixelsForFullRange));
    });
    el.addEventListener('pointerup', function (e) {
      if (!dragging) { return; }
      dragging = false;
      dspark.endEdit(p.id);
      el.releasePointerCapture(e.pointerId);
    });
    el.addEventListener('dblclick', function () { dspark.setParam(p.id, p.def); });
    el.addEventListener('wheel', function (e) {
      e.preventDefault();
      var step = (p.max - p.min) / (e.shiftKey ? 400 : 80);
      dspark.setParam(p.id, clamp(dspark.getParam(p.id) - Math.sign(e.deltaY) * step));
    }, { passive: false });
  }

  dspark.onReady(function (params) {
    var knob = document.querySelector('.knob');
    initKnob(knob, params.find(function (q) {
      return q.id === knob.getAttribute('data-param');
    }));
  });
})();
