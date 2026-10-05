import type { ClassId } from '@abyss/engine';

export const CLASS_COLOR: Record<ClassId, number> = {
  vanguard: 0x5b8def, arcblade: 0xf08a3c, riftweaver: 0x39c6c0, lifebinder: 0x6bd66b, chronoStalker: 0xe6c74c,
};
export const CLASS_GLYPH: Record<ClassId, string> = {
  vanguard: 'V', arcblade: 'A', riftweaver: 'R', lifebinder: 'L', chronoStalker: 'C',
};
export const css = (n: number) => `#${n.toString(16).padStart(6, '0')}`;
