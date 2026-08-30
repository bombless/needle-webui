// Compatibility shim for the fixed 120-byte Needle export header.
// Keep the mapping local to the browser parser; tensor-directory offsets are
// always beyond this range, so these exact offsets are unambiguous.
(() => {
  const original = DataView.prototype.getUint32;
  const map = new Map([[100,80],[104,84],[108,88],[112,92],[120,96],[124,100],[128,104],[132,108]]);
  DataView.prototype.getUint32 = function(offset, littleEndian) {
    return original.call(this, map.get(offset) ?? offset, littleEndian);
  };
})();
