# AvatarHelper

Blender add-on (4.0+) that refits clothing made for one avatar base onto another.

## Install

Download `AvatarHelper-<version>.zip` from the [latest release](https://github.com/Curtis-VL/AvatarHelper/releases/latest)
(don't unzip it), then in Blender: Preferences → Add-ons → *Install…* (4.2+: the ▾ menu → *Install from Disk…*),
pick the zip, and enable **AvatarHelper**. The panel is in the 3D view sidebar (N) → *AvatarHelper*.

### Updates

AvatarHelper checks for a newer release when Blender starts (in the background; turn it off in the add-on's
preferences). When there is one, the panel shows *Update available* with an **Update** button, which downloads
and installs it; restart Blender to finish. The add-on's preferences also have *Check for Updates* and a link to
the release notes. A development copy (linked to a git checkout, below) isn't updated this way: use git.

### Releases

Every push to `main` that changes the add-on publishes a release automatically
(`.github/workflows/release.yml`): an installable zip, with notes listing the changes since the previous one.
The version is `major.minor.build`: major and minor come from `bl_info["version"]` in
`avatar_helper/__init__.py` (bump them by hand for bigger releases), and build is the workflow's run number,
so every release is newer than the last without editing anything. Run the workflow by hand from the Actions
tab to release without a change. The update check reads the latest release, so it needs the repository (or
at least its releases) to be public.

## Install for development

Link the add-on folder into Blender's add-ons directory, so edits are picked up with
**F3 → Reload Scripts** and no reinstalling is needed:

```powershell
New-Item -ItemType Directory -Force "$env:APPDATA\Blender Foundation\Blender\4.0\scripts\addons"
New-Item -ItemType Junction -Path "$env:APPDATA\Blender Foundation\Blender\4.0\scripts\addons\avatar_helper" -Target "$PWD\avatar_helper"
```

Then enable **AvatarHelper** in Preferences → Add-ons. The panel is in the 3D view sidebar (N) → *AvatarHelper*.

## Workflow

### Once per base: create a conversion

A **conversion** (e.g. `Manuka > Custom`) records how one base avatar maps onto your avatar. It's saved to a
file and reused for every outfit made for that base, so the base avatar only needs importing once.

1. Import the base avatar the clothing was made for, next to your avatar.
2. Open **Create Conversion**, pick the base's **body** mesh, and your avatar's body in the main panel.
   Armatures are taken from the bodies' Armature modifiers (shown under each body); the armature field only
   appears for a body without one. *Check Bone Mapping* reports which humanoid bones were found.
3. Give it a name (defaults to `<base body> > <your body>`) and press **Create Conversion**. This matches the
   base rig's pose to yours, matches the base body onto your body, saves the result, and selects it. The base
   rig is left as it was. A warning appears if the bodies didn't match up well.
4. Optional: tick *Show Calibration* to see the result (`AH_Calibration`, hidden by default since it just
   overlaps your body). Slide *Preview* from 0→1 to morph the base onto your avatar; Viewport shading → Color →
   Attribute shows `AH_Error` (green = good, red ≥ 1 cm off). To fix a bad area, add landmark pairs in
   *Advanced* (sphere on the base feature, cube on the same feature of your avatar) and create it again, or
   sculpt the `Fitted` shape key and press *Save Calibration Edits*.

Conversions are saved in `Documents/AvatarHelper/Conversions` (change it in the add-on's preferences); the
folder button next to the conversion opens it, for sharing, backing up or deleting conversions. The base
avatar can be deleted afterwards.

A conversion is saved relative to the two armatures, as if both avatars stood at the origin, so it doesn't
matter where the base or your avatar stood while creating it, or where an outfit stands when fitting.
(Conversions made before this was added still work as before; create them again to get this.) It also
records the shape key values of both bodies (e.g. the base's `Foot_heel`): see *Shape keys* below.

### Every outfit: fit clothing

1. Import the outfit (it can come with a partial armature: only the bones it uses). Choose the conversion for
   its base, and set your avatar's body.
2. Add the outfit meshes (fitted and finalized results can't be added) and press *Auto-Sort Layers* (orders them innermost-first from how they overlap,
   keeping your order where that doesn't tell, and removes duplicates). Each item has a mode: *Auto* (shoes if
   skinned almost entirely below the knee and mostly to the feet), *Fit*, or *Shoes*, a solver, and
   optionally its own Shape Detail (see below); the box under the list shows the selected item's settings,
   and anything not set per item uses the defaults below it.
3. **Fit Clothing**. The outfit's armature is posed like the base was for the conversion, and put back
   afterwards; there's no armature to pick. Results go to the `AH_Fitted` collection as `<name>_fitted`:
   fitted meshes plus a fitted copy of each outfit armature (rest pose in your avatar's proportions). Fitting
   again replaces them. Fit warns if the list order contradicts how the items overlap.
4. **Finalize** when you're happy with it. The fitted clothing takes the originals' names and their place in
   your collections (Unity animations and toggles go by name); the originals are renamed `<name> (original)`
   and moved to a hidden `AH_Originals` collection, still there to fit again. Finalized clothing is never
   replaced by later fits, and the list is cleared for the next outfit, or the next part of this one (parts
   finalized later share the outfit armature finalized first). *Rig To* chooses what it's rigged to:
   - *Outfit Armature*: the fitted copy of the outfit's armature, to merge onto your avatar in Unity (e.g.
     Modular Avatar *Merge Armature*).
   - *Avatar Armature*: your avatar's own armature. Clothing near the body takes the body's own skin weights at
     the nearest point of the skin, so in any pose it bends exactly like the skin under it, which keeps
     clipping to a minimum (in tests about the same or less than the outfit armature, and much less for tight
     clothing). Loose parts keep the outfit's weights on your avatar's matching bones, and the outfit's own
     bones (skirt or string bones for physics) are added to your avatar's armature under the matching bone.
     At most 4 bones per vertex, as Unity uses. Your avatar's armature must be in its rest pose.

### Shape keys

A conversion only fits clothing shaped for the base as it was when the conversion was made. So when fitting,
clothing shape keys with the same name as one of the base body's (ignoring case, spaces and underscores; e.g.
`Foot_heel`) are set to the value the base had, and the fitted copy keeps that value; the original is left as
it was. Fit lists any keys it changed, and warns if your avatar's own shape keys changed since the conversion
was made (create it again then).

### Touching up

Bumps on a fitted garment are most visible along its edges (a hem, a cuff, a hood's rim). To fix one,
select the fitted mesh, enter Edit Mode, select the bumpy part (generously, e.g. the whole hem band) and
press **Touch Up → Restore Original Shape**. The selection is rebuilt from the original garment's shape:
only how it was stretched and turned on a larger scale is kept from the fit, while the rest of the garment
stays put and holds it in place. In the redo panel:

- *Shape Size* (5 cm): the selection keeps the original's shape up to this size. Raise it if bumps remain,
  lower it if the part stops following the body.
- *Falloff* (1 cm): blends into the unselected part, so there's no seam.
- *Keep Outside Body*: points that were outside the avatar's body stay outside it (by up to Min Gap). Parts
  tucked inside the body by design are left as fitted.

It works on whichever shape key you're editing, comparing with the original's same key (e.g. a "crop" key).

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

## How it works

- **Bone mapping**: side comes from the last L/R marker in a name (so `Index Finger Left 02.R` is right-hand),
  and fingers always belong to the hand they hang off; finger-like names elsewhere (a harness's
  `Spine Ring L`) are accessory bones and ignored.
- **Conversions** store the base body before and after calibration, the base's joint positions (at rest and
  matched), how each humanoid bone was turned, and the base's foot frames. At fit time they're lined up with
  each outfit armature by its joints (a similarity fit that drops joints which disagree, so a moved or
  mis-named bone can't skew it) and with your avatar the same way, so neither needs to be where it was.
- **Calibration**: joint positions (centre-line joints snapped onto the rig's mirror plane, since some rigs put
  e.g. the Hips head off-centre), foot tips (and hand tips on rigs without finger bones), and landmarks drive
  an RBF space warp; then a non-rigid ICP pass pulls the warped source body onto the target surface through a
  membrane that is stiff first and loose last. Unmatched areas follow their matched neighbours.
- **Pose matching** (part of Create Conversion) also uses the mesh foot tip when a rig has no toe bones, so
  heel-shaped feet get rotated flat.
- **Shoes** are not warped. Each shoe is taken as designed (source rest pose, so heels stay heels), turned to
  face along the target foot, scaled uniformly to its length and width (growing up to 15% more to keep ~3 mm
  of room around the foot), ankle over ankle, and its sole set on the target's floor. Layers listed before the
  shoes (socks, stockings) are then tucked inside them, never into the body.
- **Fitting**: clothing is fitted exactly as displayed, including active shape keys (e.g. `Foot_heel`); keys are
  kept and still work afterwards. Every vertex is moved by a smooth RBF field built from the body displacements,
  so a hoodie's 3 cm gap stays ~3 cm. Near the skin this is blended with a surface binding (position in a frame
  at the nearest surface point, using smoothly interpolated normals so the body's polygons aren't pressed into
  tight clothing). Displacements (not positions) are smoothed, so folds survive.
- **Solver**: by default (*Shape-preserving*) the positions above are only suggestions. From them the tool
  estimates how the fabric should stretch and turn around each vertex, smooths that over *Shape Detail* (1 cm),
  and solves for the whole garment so its original edges, stretched and turned that way, fit together. Detail
  smaller than Shape Detail (cables, folds, seams) comes from the original garment, so suggestions that disagree
  can't produce bumps, spikes or crumples. Surfaces that share no edges (a hem's lining, a strap's two sides, a
  ring and its strap loop) are linked so they stay stacked. Collisions and tucks become constraints in the
  solve, so each correction spreads smoothly. *Direct* places each vertex on its own instead (the older method).
- **Rigid parts** (small separate pieces, or the `AH_Rigid` group) move without deforming; touching pieces such
  as gems set in a pendant move as one. They take their position and size from the material around them, but
  their rotation from the smooth warp, so a pendant resting in a cleavage doesn't spin to face sideways.
- **Layers**: where an item rests on a lower layer (or is tucked under it) and is closer to it than to the body,
  it rides on that layer (anchored to its surface the same way clothing near the skin is anchored to the body),
  so a shirt over trousers keeps its exact overlap instead of both being fitted to the body separately and
  drifting into each other. Items that are only near a layer, like a choker inside a shirt collar, keep
  following the body.
- **Only new crossings are fixed**: an item is pushed out of a lower layer only where it was outside it on the
  source and crossed it during fitting, and only for overlaps up to 1 cm deep (clothing is thin; deeper means it
  has reached the other side). The body, being solid, is checked 3 cm deep. "Under" a layer only counts for
  surfaces facing away from the body, so the back of a chain or pendant isn't mistaken for being tucked in.
- **Original gaps are kept**: each point is pushed back to its own gap from the source (capped at Min Gap), so a
  tight knit stays as tight as designed instead of parts of it being lifted to 2 mm, which reads as wobble.
- **Hidden parts stay hidden**: clothing tucked into the source body (a hem curling under the waistband, a strap
  end disappearing into the thigh, a sock toe squeezed inside the foot to fit a shoe) is carried along at the
  same depth rather than pushed out, which would crumple it. This applies near open edges, or where nothing of
  the same garment lies over it; a strap that merely dips into the skin is pushed out whole, keeping it clean.
  Parts that were under a lower layer on the source (a tucked-in shirt) are not pushed out of it, and are kept
  under it; lower-layer parts that were under an item can't poke through it. Likewise, body detail that already
  showed through an item on the source (toes through socks meant to sit inside shoes) isn't forced back under it,
  which would press the target's toes into the sock.
- **Stacked pieces**: a garment made of separate pieces lying on each other (a knit over its own inner top, a
  collar over a shirt) is fitted as one mesh, and only the inner piece touches the body. Where the body pushes
  that inner piece out (a fuller back, detail the smooth fit doesn't follow), the push is passed on to the
  pieces lying on it, so they move out together and keep their gap instead of the inner piece coming through.
  Elsewhere each piece keeps its own shape. Which piece is over which is read from the source; a few points
  the other way round (at a folded edge) are ignored.
- **What counts as body**: avatars often have hair, accessories or clothing merged into the body mesh. Only
  the connected parts of the avatar mesh lying mostly on the calibrated body (body, head) are collided with;
  everything else is ignored, so a hood lying in long hair isn't pushed out of every strand.
- **Collision** only pushes outward: clothing vertices out of the body and lower layers, and clothing faces that
  body detail pokes through between vertices. Inner shells of thick clothing are left alone. Each push also moves
  what lies within 6 mm along the push direction, so the two sides of a strap or shell move together instead of
  being squashed flat, even when they share no edges (e.g. garters modelled sinking into the source's skin);
  sideways along the surface it fades smoothly over ~1 cm, giving a gentle bump rather than a lump.

## Tests

```powershell
& "C:\Program Files\Blender Foundation\Blender 4.0\blender.exe" --background --factory-startup --python tests/run_tests.py
```

Builds synthetic avatars (A-pose source → T-pose target, longer legs, larger chest) with a shirt and hoodie,
creates a conversion (into `tests/output/conversions`), fits, fits again with the base avatar deleted, and
checks gap preservation, penetration and layering. It saves
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
- Pieces of one garment only move together where the body pushes the lower one out (see *Stacked pieces*);
  there's no general self-collision, so pieces that drift into each other for other reasons aren't separated.
