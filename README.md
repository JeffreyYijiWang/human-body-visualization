# Slices of Meat?

### *MPR Multi-Volume Full FX Objects v23 UI Curve Panels*

![](images/hero_image.png)
<br>

**Hero image placeholder:** a full-screen view of the software showing a colored anatomical/meat volume, a curved slicing plane, and the side curve inspection panels.

<table>
  <tr>
    <td align="center" width="100%">
      <img src="gifs/hero_demo.gif" width="850"><br>
    </td>
  </tr>
  <tr>
    <td align="left" width="100%">
      <strong>Hero video placeholder.</strong> Suggested capture: orbit through the color volume, switch to multi-volume view, open the curved-plane editor, and hide/show the UI for a clean exhibition-style shot.
    </td>
  </tr>
</table>

---

## Abstract

**Slices of Meat?** is an experimental visualization system for navigating stacked anatomical/meat imagery as a spatial, sculptural, and cultural object. Built around `mpr_multivolume.py`, the project loads a main color volume alongside optional skeleton and gradient-distance volumes, then allows the viewer to cut, bend, compare, mask, transform, and recompose the body through a set of interactive viewing modes.

The work borrows the visual language of medical imaging, butcher diagrams, taxidermy display, projection mapping, and computational image processing. Instead of treating anatomical visualization only as a path toward accuracy, the tool asks what happens when slices become material: hung on walls, packed into grids, inflated, curved, reflected, blocked, or reorganized into strange views of flesh.

The project sits between **scientific instrument**, **image-making system**, and **speculative exhibition interface**. It is both a viewer and a way of staging the body.

---

## Project statement

The original ambition of many human visualization projects was to make the body legible. The body was dissected, photographed, scanned, registered, segmented, and reconstructed so that medical knowledge could become more precise. The **Visible Human Project**, for example, made public cross-sectional cryosection, CT, and MRI images of human bodies as a reference for anatomy, medical imaging, and computational research.

My project begins from that lineage, but moves toward a different question:

> **What if the goal is not only to see the body correctly, but to see how many cultural and computational forms a body can take?**

In this viewer, a slice is not just a medical cross-section. It can become a butcher cut, a projection, a specimen, a trophy, a wall object, a screen surface, or a distorted memory of a body. The tool treats volume data as an unstable material: one that can be opened, folded, recomposed, and misread.

I was thinking about the difference between a body used for science and a body used as a display object. A taxidermy moose head on a wall is not only an animal; it is a token of hunting, ownership, conquest, memory, decoration, and control. A medical volume is also a form of display. It claims a different purpose—education, accuracy, diagnosis—but it still transforms a body into an object that can be viewed, rotated, sliced, and possessed through vision.

This project explores the uneasy space between those two forms of looking.

---

## Motivation and visual research

The project grew from several overlapping questions:

- What happens when pieces of meat are treated like landscapes or architectural sections?
- What is the difference between a scientific specimen, a butchered animal, and a trophy on a wall?
- How do farmed bodies, genetically modified animals, artificial inflation, steroids, plastic surgery, and cosmetic alteration complicate the idea of a “natural” body?
- What does it mean to stack slices of flesh into a volume and then ask a computer to find the best view?
- Can a medical visualization interface become an image-making system instead of only an anatomy tool?

The resulting software creates a space where the user can move through the volume, compare tissue-like structures, generate strange projections, and search for views that feel visually or conceptually charged.

<table>
  <tr>
    <td align="center" width="50%">
      <img src="images/inspiration_medical_archive.png" width="420"><br>
    </td>
    <td align="center" width="50%">
      <img src="images/inspiration_taxidermy_meat.png" width="420"><br>
    </td>
  </tr>
  <tr>
    <td align="left" width="50%">
      <strong>Placeholder:</strong> medical visualization, cryosection images, anatomy atlases, DICOM viewers, Visible Human / Visible Korean references.
    </td>
    <td align="left" width="50%">
      <strong>Placeholder:</strong> butcher charts, taxidermy, farm animals, artificial bodies, bio-art, hybrid creatures, meat display.
    </td>
  </tr>
</table>

---

## Conceptual references

### Visible bodies

The project is indebted to the history of digital anatomical datasets and sectioned-image projects. The **Visible Human Project** transformed cadaveric bodies into complete digital image volumes. **Visible Korean** and related visible-body projects extend this lineage through true-color, high-resolution sectioned images used for anatomical research, education, and virtual human models.

These projects make the body available as data. My project asks how that data can also become an aesthetic and critical material.

### Video cubes, time cubes, and slicing surfaces

The project also connects to computational artworks and visualization systems that treat image sequences as volumes. **Video Cubism** allowed users to slice through a video cube using arbitrary planes and curved surfaces. **Stylized Video Cubes** treated video as a space-time volume for non-photorealistic rendering. **Khronos Projector** turned touch into a way of deforming time inside a video surface.

Those projects helped me think about the slice not as a fixed medical convention, but as an interface: a moving surface that can cut through data, time, memory, and image.

### Hybrid bodies and speculative flesh

Artists such as **Francesco Albano** and **Patricia Piccinini** influenced the project’s interest in flesh that appears altered, synthetic, tender, grotesque, or engineered. Piccinini’s hybrid creatures are especially relevant because they are simultaneously biological, artificial, vulnerable, and designed. They make the viewer question what kind of body they are looking at—and what obligations that act of looking creates.

---

## The software as an exhibition interface

`mpr_multivolume.py` is not only a program for inspecting a volume. It is a small studio for producing images from a body-like dataset.

The interface supports:

- **single slice viewing** for basic inspection,
- **axis and local oblique views** for moving through the volume from multiple directions,
- **multi-volume comparison** between color, gradient-distance, and skeleton data,
- **curved-plane viewing** where a slice becomes a bendable surface,
- **side curve inspection panels** for reading the curve from U, V, and surface views,
- **3D object interventions** that mask, block, reflect, or shift the slice,
- **screen-space transformations** that reorganize the image,
- **timeline controls** for recording paths and revisiting views,
- and **heuristics** that estimate visual qualities such as filled area, blob count, and interest score.

The interface becomes a kind of dissection table, projection surface, and editing desk at the same time.

<table>
  <tr>
    <td align="center" width="100%">
      <img src="images/interface_overview.png" width="850"><br>
    </td>
  </tr>
  <tr>
    <td align="left" width="100%">
      <strong>Placeholder:</strong> annotated overview of the software interface showing the main slice, side curve panels, top toolbar, object controls, timeline, and heuristic panel.
    </td>
  </tr>
</table>

---

## A key technical problem: making the slice behave like a surface

A flat MPR viewer is relatively direct: choose a point in the volume, choose two axes for the image plane, sample the voxel data, and display the result. But a curved plane is less obvious. It is not just a rectangle passing through the volume. It is a surface that bends away from its own base plane.

The curved-plane editor solves this by treating the slice as a parameterized surface. Each screen pixel maps to a local coordinate on the plane, and that coordinate is displaced along the plane normal by a curve function.

```text
screen pixel
    ↓
local coordinates (u, v)
    ↓
base plane position
    ↓
curve displacement along normal
    ↓
3D volume coordinate
    ↓
sampled voxel color
```

In practical terms, the system turns a flat image plane into a deformable probe. This matters conceptually because the viewer no longer simply cuts the body: the viewer presses, bends, and reshapes the surface of vision.

<table>
  <tr>
    <td align="center" width="100%">
      <img src="images/curved_plane_diagram.png" width="800"><br>
    </td>
  </tr>
  <tr>
    <td align="left" width="100%">
      <strong>Diagram placeholder:</strong> show a volume cube, a flat slicing plane, then a curved slicing plane displaced along the normal. Label U, V, normal, amplitude, and radius.
    </td>
  </tr>
</table>

---

## Development note: from tool to image system

A major development challenge was keeping the viewer fast while the project became more visually ambitious. Early versions relied heavily on CPU-side sampling and image construction. This was reliable but slow, especially when drawing multiple panels or applying effects. Later versions moved toward a more consistent GPU compositor: slices are rendered into offscreen textures and then composed into the final display.

This shift changed the structure of the project. Instead of thinking of the software as a single viewer, I began to think of it as a series of passes:

```text
volume data → slice sampling → panel composition → image effects → UI / capture
```

That pipeline made the project easier to extend. It also clarified the artistic structure: every view is a constructed image, not simply a transparent window into data.

---

## Gallery of visualizations

Replace these placeholders with final captures from the `images/` and `gifs/` folders.

### Multi-volume comparisons

<table>
  <tr>
    <td align="center" width="33%"><img src="images/gallery_multivolume_color.png" width="260"></td>
    <td align="center" width="33%"><img src="images/gallery_multivolume_gradient.png" width="260"></td>
    <td align="center" width="33%"><img src="images/gallery_multivolume_skeleton.png" width="260"></td>
  </tr>
  <tr>
    <td align="left">Color volume as flesh-like image.</td>
    <td align="left">Gradient-distance view as thickness / density cue.</td>
    <td align="left">Skeleton view as extracted internal scaffold.</td>
  </tr>
</table>

### Curved slicing and projection

<table>
  <tr>
    <td align="center" width="50%"><img src="images/gallery_curved_surface_1.png" width="400"></td>
    <td align="center" width="50%"><img src="images/gallery_curved_surface_2.png" width="400"></td>
  </tr>
  <tr>
    <td align="left">A curved surface cutting through the volume.</td>
    <td align="left">A side-panel view showing the curved section as a spatial surface.</td>
  </tr>
</table>

### Butchered grids / reorganized flesh

<table>
  <tr>
    <td align="center" width="33%"><img src="images/gallery_grid_sort_1.png" width="260"></td>
    <td align="center" width="33%"><img src="images/gallery_cut_fx_1.png" width="260"></td>
    <td align="center" width="33%"><img src="images/gallery_blob_pack_1.png" width="260"></td>
  </tr>
  <tr>
    <td align="left">Pixel or slice sorting by value, hue, area, or density.</td>
    <td align="left">Cut and drift effects that separate the image into meat-like fragments.</td>
    <td align="left">Packed blobs as a speculative butcher chart or wall arrangement.</td>
  </tr>
</table>

### Object interventions

<table>
  <tr>
    <td align="center" width="50%"><img src="images/gallery_object_masker.png" width="400"></td>
    <td align="center" width="50%"><img src="images/gallery_object_reflector.png" width="400"></td>
  </tr>
  <tr>
    <td align="left">A 3D object masking a region of the sampled body.</td>
    <td align="left">A reflector/shifter object altering the visible slice.</td>
  </tr>
</table>

### Timeline and capture

<table>
  <tr>
    <td align="center" width="50%"><img src="gifs/gallery_timeline_path.gif" width="400"></td>
    <td align="center" width="50%"><img src="images/gallery_capture_sheet.png" width="400"></td>
  </tr>
  <tr>
    <td align="left">Timeline playback through a recorded camera path.</td>
    <td align="left">Captured frames sorted from largest filled area to smallest.</td>
  </tr>
</table>

---

Run the project:

```bash
python mpr_multivolume.py
```

The script expects `.npy` volume files for the main color volume and optional auxiliary volumes. Update the paths in the script to match your local dataset.

---

## Controls / interaction guide

### Core keys

| Key | Action |
| --- | --- |
| `T` | Cycle view mode |
| `H` | Hide/show mouse cursor |
| `Y` | Toggle analysis / heuristics so blob counters do not run every frame |
| `F1` | Hide/show UI |
| `F2` | Hide/show gizmo |
| `F3` | Recommend or apply an interesting next view |
| `F4` | Start/stop 24 fps capture |
| `F5` | Toggle blob debug overlay |
| `F6` | Search for blob-dense / low-interest view |
| `F7 / F8` | Cycle color filter mode / target |
| `F9` | Cycle display variant |
| `F10` | Toggle aux-from-main mode |
| `F11` | Cycle frame transform |
| `F12` | Toggle Brownian auto motion |

### UI tabs

| Tab | Purpose |
| --- | --- |
| **Move / Brush** | Navigation, slicing, cursor and brush-like interaction |
| **Timeline** | Waypoints, playback, camera paths, looping |
| **Screen FX** | Image-space transformations and fragment effects |
| **Objects** | Add and edit blockers, maskers, reflectors, and shifters |
| **Plane** | Curved-plane settings, amplitude, radius, side views |
| **Heuristics** | Blob count, fill area, interest score, and debugging |

For exhibition display, use **Hide UI** or `F1` to move between the working interface and a cleaner projected image.

---

---

## References

### Anatomical and medical visualization

1. National Library of Medicine. **Visible Human Project**.  
   https://www.nlm.nih.gov/research/visible/visible_human.html

2. National Library of Medicine. **NLM Digital Projects**.  
   https://www.nlm.nih.gov/digitalprojects.html

3. Kim, Chung Yoh; Chung, Min Suk; Park, Jin Seo. **Visible Korean based on true color sectioned images for making realistic digital human, twenty years’ record: a review.** *Surgical and Radiologic Anatomy*, 2024.  
   https://pubmed.ncbi.nlm.nih.gov/38717503/

4. Chung, Beom Sun; Park, Jin Seo. **Real-Color Volume Models Made from Real-Color Sectioned Images of Visible Korean.** *Journal of Korean Medical Science*, 2019.  
   https://pmc.ncbi.nlm.nih.gov/articles/PMC6417999/

5. OsiriX. **DICOM Viewer.**  
   https://www.osirix-viewer.com/

6. IMAIOS. **Visible Human Project: normal anatomy / e-Anatomy.**  
   https://www.imaios.com/en/e-anatomy

### Video volumes, projection, and computational image systems

7. Fels, Sidney; Mase, Kenji. **Interactive Video Cubism.** NPIV, 1999.  
   https://www.sciweavers.org/node/164045

8. Klein, Allison W.; Sloan, Peter-Pike J.; Finkelstein, Adam; Cohen, Michael F. **Stylized Video Cubes.** SCA, 2002.  
   https://doi.org/10.1145/545261.545264

9. Cohen, Michael F.; Colburn, Alex; Drucker, Steven. **Image Stacks.** Microsoft Research Technical Report, 2003.  
   https://www.microsoft.com/en-us/research/publication/image-stacks/

10. Cassinelli, Alvaro. **KHRONOS PROJECTOR.** 2004.  
    https://alvarocassinelli.com/khronos-projector/

### Artistic and conceptual references

11. Francesco Albano. **Selected works.**  
    https://medinart.eu/works/francesco-albano/

12. Queensland Art Gallery | Gallery of Modern Art. **Looking at Patricia Piccinini’s monsters looking at us.**  
    https://www.qagoma.qld.gov.au/stories/looking-at-patricia-piccininis-monsters-looking-at-us/

13. Thoughts Become Words. **Curious Affection: Hybrids of Patricia Piccinini’s Biotechnology Art.**  
    https://thoughtsbecomewords.com/2018/07/22/curious-affection-hybrids-of-patricia-piccininis-biotechnology-art/

14. Jason Webb. **Morphogenesis Resources.**  
    https://github.com/jasonwebb/morphogenesis-resources

---

## Closing note

This project began as a viewer, but it became a question about looking. Medical visualization often promises access to the inside of the body. This work accepts that promise, then bends it. It asks what happens when the inside is not only studied, but arranged, projected, exaggerated, sorted, and displayed.

The body becomes data. The data becomes a surface. The surface becomes an image. The image becomes a specimen again.
