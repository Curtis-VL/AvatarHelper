# AvatarHelper

Blender add-on (4.0+) that refits clothing made for one avatar base onto another.

## Install for development

Link the add-on folder into Blender's add-ons directory, so edits are picked up with
**F3 → Reload Scripts** and no reinstalling is needed:

```powershell
New-Item -ItemType Directory -Force "$env:APPDATA\Blender Foundation\Blender\4.0\scripts\addons"
New-Item -ItemType Junction -Path "$env:APPDATA\Blender Foundation\Blender\4.0\scripts\addons\avatar_helper" -Target "$PWD\avatar_helper"
```

Then enable **AvatarHelper** in Preferences → Add-ons. The panel is in the 3D view sidebar (N) → *AvatarHelper*.

## Workflow

1. **Set up** – import the source avatar (the one the outfit was made for), the outfit, and the target avatar.
   Pick the source/target body meshes (armatures are auto-detected from their modifiers).
   *Check Bone Mapping* reports which humanoid bones were found.
2. **Match Pose** – rotates the source rig (and the outfit's own armature) so limbs point the same way as the target's.
3. **Calibrate** – produces `AH_Calibration`: the source body with a `Fitted` shape key.
   - Slide *Preview* from 0→1: the source body should morph onto the target body.
   - Viewport shading → Color → Attribute shows `AH_Error` (green = good, red ≥ 1 cm off).
   - Bad area? Add landmark pairs (sphere on the source feature, cube on the same feature of the target) and recalibrate,
     or sculpt the `Fitted` shape key directly.
4. **Fit Clothing** – add the outfit meshes and press *Auto-Sort Layers* (orders them innermost-first from how
   they overlap on the source avatar, keeping your order where the source doesn't tell, and removes duplicates).
   Each item has a mode: *Auto* (shoes if skinned almost entirely below the knee and mostly to the feet),
   *Fit*, or *Shoes*. Then *Fit Clothing*. Results go to the `AH_Fitted` collection: fitted meshes plus a fitted
   copy of each outfit armature (rest pose in the target's proportions), ready for Modular Avatar
   *Merge Armature* in Unity. Fit warns if the list order contradicts the source.

### Controls that matter most

| Setting | Effect |
|---|---|
| Skin Follow / Blend Range | Clothing this close to the skin follows body detail exactly; further out it uses the smooth warp and keeps its shape. |
| Field Resolution / Smoothing | Detail vs smoothness of the warp for loose clothing. |
| Rigid Part Size, `AH_Rigid` vertex group | Parts that move without deforming (buttons, buckles, armour plates). |
| Min Gap / Collide With Lower Layers | Outward-only push to clear the body and the layers listed above. |

## How it works

- **Bone mapping**: side comes from the last L/R marker in a name (so `Index Finger Left 02.R` is right-hand),
  and fingers always belong to the hand they hang off.
- **Calibration**: joint positions (centre-line joints snapped onto the rig's mirror plane, since some rigs put
  e.g. the Hips head off-centre), foot tips (and hand tips on rigs without finger bones), and landmarks drive
  an RBF space warp; then a non-rigid ICP pass pulls the warped source body onto the target surface through a
  membrane that is stiff first and loose last. Unmatched areas follow their matched neighbours.
- **Match Pose** also uses the mesh foot tip when a rig has no toe bones, so heel-shaped feet get rotated flat.
- **Shoes** are not warped. Each shoe is taken as designed (source rest pose, so heels stay heels), turned to
  face along the target foot, scaled uniformly to its length and width (growing up to 15% more to keep ~3 mm
  of room around the foot), ankle over ankle, and its sole set on the target's floor. Layers listed before the
  shoes (socks, stockings) are then tucked inside them, never into the body.
- **Fitting**: clothing is fitted exactly as displayed, including active shape keys (e.g. `Foot_heel`); keys are
  kept and still work afterwards. Every vertex is moved by a smooth RBF field built from the body displacements,
  so a hoodie's 3 cm gap stays ~3 cm. Near the skin this is blended with a surface binding (position in the
  nearest triangle's frame). Displacements (not positions) are smoothed, so folds survive.
- **Layers**: where an item rests on a lower layer (or is tucked under it) and is closer to it than to the body,
  it rides on that layer (anchored to its surface the same way clothing near the skin is anchored to the body),
  so a shirt over trousers keeps its exact overlap instead of both being fitted to the body separately and
  drifting into each other. Items that are only near a layer, like a choker inside a shirt collar, keep
  following the body.
- **Only new crossings are fixed**: an item is pushed out of a lower layer only where it was clear of it on the
  source and crossed it during fitting, and only for overlaps up to 1 cm deep (clothing is thin; deeper means it
  has reached the other side). The body, being solid, is checked 3 cm deep.
- **Hidden parts stay hidden**: clothing tucked into the source body (a hem curling under the waistband, a strap
  end disappearing into the thigh, a sock toe squeezed inside the foot to fit a shoe) is carried along at the
  same depth rather than pushed out, which would crumple it. This applies near open edges, or where nothing of
  the same garment lies over it; a strap that merely dips into the skin is pushed out whole, keeping it clean.
  Parts that were under a lower layer on the source (a tucked-in shirt) are not pushed out of it, and are kept
  under it; lower-layer parts that were under an item can't poke through it. Likewise, body detail that already
  showed through an item on the source (toes through socks meant to sit inside shoes) isn't forced back under it,
  which would press the target's toes into the sock.
- **Collision** only pushes outward: clothing vertices out of the body and lower layers, and clothing faces that
  body detail pokes through between vertices. Inner shells of thick clothing are left alone. Each push also moves
  everything within 6 mm of that point (fading over the next 1 cm), so the two sides of a strap or shell move
  together instead of being squashed flat, even when they share no edges (e.g. garters modelled sinking into
  the source's skin).

## Tests

```powershell
& "C:\Program Files\Blender Foundation\Blender 4.0\blender.exe" --background --factory-startup --python tests/run_tests.py
```

Builds synthetic avatars (A-pose source → T-pose target, longer legs, larger chest) with a shirt and hoodie,
runs the full pipeline, and checks gap preservation, penetration and layering. It saves
`tests/output/demo_start.blend` (ready to click through) and `demo_result.blend`.

## Known limitations / next steps

- Objects parented directly to a bone (rather than skinned) are not repositioned correctly yet.
- Topology-changing modifiers (Mirror, Subdivision) on clothing must be applied or disabled before fitting.
- Shape key deltas are carried over unrotated; keys on limbs that change angle a lot may need touch-up.
- Collision assumes outward-facing normals on clothing.
- Clipping that already exists on the source (clothing inside the source body) is kept as it was.
- A heeled shoe on a flat target foot can't contain the foot: some foot shows through (hide it with a shape key or
  toggle, as usual for shoes).
- Min Gap (2 mm) also applies to very thin layers such as stockings; lower it to ~1 mm for a tighter look.
- Body blendshape transfer (target breast/hip sliders driving the outfit) is not implemented yet.
- Calibrations are per scene; exporting/importing them as shareable profiles is planned.
