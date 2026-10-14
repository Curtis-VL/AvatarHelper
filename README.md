# AvatarHelper

Blender add-on (4.0+) that refits clothing made for one avatar base onto another.

**Note:** At the moment, no conversions are included, so you'll need both the base of your avatar and the one the clothing is made for to create a conversion. Conversions may need to be re-created after some updates that alter the calibration, be sure to keep a .blend file on hand for to make this quick and easy!

I'll include conversions with a future update so there's no need to create conversions manually for commonly used bases. ^^

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

### Once per base pair: Create a conversion

<img width="2106" height="1208" alt="1" src="https://github.com/user-attachments/assets/9049caba-420d-4aa1-bb37-6108ea4ca576" />

Step by step:
- Import both bases into the scene. (Armature is hidden in the image for clarity, do not remove the armatures!)
- Ensure the body mesh of both avatars is **only** the body or body+face, it should be the same on both bases.
- Select the avatar to fit to, and the base avatar as shown in the image above.
- Enter a name and click 'Create Conversion'.

For fitting clothes from the same avatar at a later date, you won't need to go through this process again.

### Every outfit: Fit clothing

<img width="2106" height="1208" alt="2" src="https://github.com/user-attachments/assets/e736a7ed-32ea-4099-9d8d-d51076e7a1b4" />

Step by step:
- Import the outfit for an avatar **you've already setup a conversion for with the above steps.**
- Select the body mesh for the avatar, the same one you used for the calibration. (Ensure it is the same shape as when you calibrated, so body or body+face - Make sure nothing like accessories or hair is part of the mesh)
- Add clothing meshes to the 'Layers' list, starting with the inner-most (Closest to the skin) and working your way outwards.
- Click 'Fit Clothing'. (Clicking this again will override un-finalized fitted clothing, so you can change settings and run it again to try different fits)

**Note:** If your outfit has multiple variants, fit one variant first then the next one after finalizing the first. (For example, short and long sleeves, crop top or full length top)

**Limitation:** Right now, shoes cannot be fitted, head wear may also have mixed results. The add-on will scale them to be roughly in the right location and size, but you'll need to do some manual dragging around with proportional editing to get them to fit correctly. (This is usually pretty quick!)

Basic troubleshooting:
- If an area appears bumpy or warped, select the verticies of that around and a generous area around it, then click 'Restore Original Shape'.
- If some areas still look incorrect, you can try changing the Solver from Default to 'Shape Preserving' and setting the 'Own Shape Detail' for the affected mesh higher, then click 'Fit Clothing' again. (0.05 is usually a good amount)
- For some clothing such as a neckless or choker, using the 'Direct' Solver mode seems to show better results, then press 'Fit Clothing' again.

**Happy with the final result?**

Click 'Finalize', this will merge it with your avatar armature and also merge the weights!

The final result:

<img width="480" height="527" alt="3" src="https://github.com/user-attachments/assets/2b247c5e-c928-4f0c-841c-0869d393aa66" />

(Manuka > Selestia in this case)

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
