# AvatarHelper

Blender add-on (4.0+) that refits clothing made for one avatar base onto another.

## Install

Download `AvatarHelper-<version>.zip` from the [latest release](https://github.com/Curtis-VL/AvatarHelper/releases/latest)
(don't unzip it)

In Blender: Edit → Preferences → Add-ons → *Install…* (4.2+: the ▾ menu → *Install from Disk…*),
pick the zip, and enable **AvatarHelper**.

The panel is in the 3D view sidebar (N key) → *AvatarHelper*.

## Reporting bugs

**Report a Bug**, at the bottom of the AvatarHelper sidebar (and in the add-on's preferences), opens the
[issues page](https://github.com/Curtis-VL/AvatarHelper/issues) and copies your AvatarHelper, Blender and system
versions to the clipboard, ready to paste into the report.

When able, please attach an example .blend file that shows the issue. This scene should include the original clothing, the intended model for the clothing, and the base model you're fitting the clothes to.

If you'd prefer not to share this publicly, feel free to share this with me on Discord!
@curtisvl

## Workflow

### Once per base: create a conversion

To be written

### Every outfit: fit clothing

To be written

### Shape keys

A conversion only fits clothing shaped for the base as it was when the conversion was made. For example, if the conversion was made with a 'heel foot' shape key active, the add-on will try to match this for the clothing. If the clothing doesn't have the 'heel foot' shapekey that the conversion was created with, it may lead to some unexpected results!

When clicking 'Fit Clothing', whatever shape key was in use on the outfit at this time will be the one that's fitted. You may find that setting these shape keys back to 0 after fitting causes some issues.

Try to keep this in mind. ^^

### Touching up

Bumps on a fitted clothes are most visible along its edges (a hem, a cuff, a hood's rim). To fix one,
select the fitted mesh, enter Edit Mode, select the bumpy part (generously, e.g. the whole hem band) and
press **Touch Up → Restore Original Shape**.

The selection is rebuilt from the original clothes's shape, just scaled to meet the new avatar's size.

For tighter fighting clothing (And something like a choker or neckless), try setting the Solver to 'Direct', this may lead to some better results.

For loose clothes, you can try increasing the 'Shape Detail', 0.05 seems to be a good value. This may introduce some clipping however.

Some other settings to consider:
- *Shape Size* (5 cm): the selection keeps the original's shape up to this size. Raise it if bumps remain,
  lower it if the part stops following the body.
- *Falloff* (1 cm): blends into the unselected part, so there's no seam.
- *Keep Outside Body*: points that were outside the avatar's body stay outside it (by up to Min Gap). Parts
  tucked inside the body by design are left as fitted.

### Controls that matter most

| Setting | Effect |
|---|---|
| Solver (per item) | *Shape-preserving* keeps the garment's own surface smooth (best for most fabric); *Direct* follows the body most closely (can suit chains and necklaces). |
| Shape Detail (per item, or the default) | Shape-preserving only. Higher (e.g. 5 cm) keeps more of the original garment's own shape and looks better on knits and loose clothing; skin-tight items (body suits, fitted shirts) follow the body less closely and can clip a little more, so give those 1 cm. The default is 1 cm. |

In *Advanced* (under Fit Clothing):

| Setting | Effect |
|---|---|
| Skin Follow / Blend Range | Clothing this close to the skin follows body detail exactly; further out it uses the smooth warp and keeps its shape. |
| Field Resolution / Smoothing | Detail vs smoothness of the warp for loose clothing. |
| Rigid Part Size, `AH_Rigid` vertex group | Parts that move without deforming (buttons, buckles, armour plates). |
| Min Gap / Collide With Lower Layers | Outward-only push to clear the body and the layers listed above. |

## Known limitations / next steps

- Shoes will not be fitted correctly. They will be scaled to the right size, but will need some manual adjustment!
- Items attached to the head may not be in the correct location or may be warped.
- Bras/bikini tops may not be in the perfect location and may need moving up or down a little manually.

## Need more help?

Feel free to message me on Discord. ^^
@curtisvl
