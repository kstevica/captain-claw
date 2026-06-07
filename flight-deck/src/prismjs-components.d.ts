// Prism ships language grammars as side-effect modules that @types/prismjs
// doesn't declare individually. They register themselves on the Prism global.
declare module 'prismjs/components/*'
