# human-body-visualization

![](images/hero_image.png)
<br>

**An interactive volume-slicing and visualization environment for exploring the body as image, specimen, object, and projection.**

This project centers on the script **`mpr_multivolume_full_fx_objects_v23_ui_curve_panels.py`**, an experimental viewer for navigating a 3D volume through multiple kinds of slices, transformed views, curve-based cutting planes, multi-volume comparisons, object-based interventions, and image-space effects. It combines a technical interest in medical/scientific visualization with a conceptual interest in meat, anatomy, taxidermy, spectacle, and the politics of displaying bodies.

At a basic level, the tool lets you:

- load a **main color volume**, plus optional **skeleton** and **gradient/distance** volumes,
- inspect the data through **single**, **axis**, **local oblique**, **multi-volume**, and **curved-plane** views,
- apply **screen-space FX** and image transformations,
- place **3D objects** that act like blockers, maskers, reflectors, or shifters,
- record and play back **camera waypoints/timelines**,
- and analyze slices using lightweight **heuristics** such as filled area, blob count, and flesh/bone-like balance.

<table>
  <tr>
    <td align="center" width="100%">
      <img src="gifs/hero_demo.gif" width="800"><br>
    </td>
  </tr>
  <tr>
    <td align="left" width="100%">
      Hero video / GIF placeholder. Replace with a short capture showing the main viewer, curved-plane mode, and multi-volume comparison.
    </td>
  </tr>
</table>

---

## Project overview

This work began as part of a broader **human visualization project**. The original intent of many anatomical visualization systems—such as the Visible Human Project and related medical imaging tools—was to improve medical knowledge by making the body viewable, measurable, and navigable. My interest shifts that goal slightly. Instead of only asking **how to see the body correctly**, I am also asking:

- **How else can the body be seen?**
- What happens when anatomical data is treated like a sculptural or cinematic material?
- What kinds of meaning emerge when we view a body through the logics of butcher diagrams, taxidermy displays, projection systems, or image stacks?
- How do scientific displays and cultural displays overlap?

In this project, slices are not only diagnostic images. They can also feel like:

- butchered sections,
- wall-mounted trophies,
- inflated or altered flesh,
- spatial projections,
- and interfaces for speculation about how bodies are represented.

The project pulls together influences from:

- **medical visualization**,
- **DICOM / visible human datasets**,
- **XR anatomy tools**,
- **video slicing / spacetime imaging**,
- **projection systems**,
- and **bio-art / hybrid-body art practices**.

---

## Motivation

I was motivated by a tension between two ways of looking at bodies.

On one side, there is the scientific and pedagogical ambition behind projects like the **Visible Human Project**: to dissect, digitize, and visualize the human body in order to improve medical knowledge and make anatomy more accessible and correct. On the other side, there is a cultural history of displaying bodies and animal forms through hunting trophies, butcher charts, taxidermy, preserved specimens, and contemporary biotech art.

That tension became especially compelling to me through a set of recurring images and questions:

- random pieces of meat and extracted forms,
- taxidermy animals on walls,
- the moose head as a display object,
- butcher sections and meat cuts,
- farmed animals versus wild animals,
- genetically modified or artificially inflated animals,
- cosmetic surgery, steroids, enhancement, and sculpted flesh,
- and the strange space between a scientific specimen and a symbolic body.

So this project is partly about **visualization**, but it is also about **framing**. It asks what changes when the body is re-sliced, projected, inflated, curved, stacked, or aestheticized.

### Inspirations

A few key inspirations include:

#### Medical / anatomical visualization

- **Visible Human Project** — National Library of Medicine / NIH Open Data Portal  
  https://www.nlm.nih.gov/research/visible/visible_human.html
- **DICOM converted images for the NLM Visible Human Project collection**
- **Visible Human Project: normal anatomy | e-Anatomy**
- **CvhSlicer 2.0: Immersive and Interactive Visualization of Chinese Visible Human Data in XR**
- **OsiriX DICOM Viewer**
- **SofaAPAPI-Unity3D - Interactive Virtual Simulation of Ultrasound**
- **Visible Korean based on true color sectioned images for making realistic digital human**

#### Artistic / conceptual references

- Francesco Albano  
  https://medinart.eu/works/francesco-albano/
- Patricia Piccinini and biotechnology / hybrid-body art  
  https://www.qagoma.qld.gov.au/stories/looking-at-patricia-piccininis-monsters-looking-at-us/  
  https://thoughtsbecomewords.com/2018/07/22/curious-affection-hybrids-of-patricia-piccininis-biotechnology-art/
- Morphogenesis resources  
  https://github.com/jasonwebb/morphogenesis-resources

#### Time / volume / slicing references

- **Stylized Video Cubes** (Michael Cohen et al., 2002)
- **Image Stacks** (2003)
- **Video Cubism** (Sidney Fels, Kenji Mase & Eric Lee, 1999)
- **KHRONOS PROJECTOR (2004)** — Alvaro Cassinelli
- **Out of Bounds** — Chris O'Shea
- map and projection logics such as **Mercator projection onto a cylinder**

<table>
  <tr>
    <td align="center" width="50%">
      <img src="images/inspiration_board_1.png" width="420"><br>
    </td>
    <td align="center" width="50%">
      <img src="images/inspiration_board_2.png" width="420"><br>
    </td>
  </tr>
  <tr>
    <td align="left" width="50%">
      Placeholder: an inspiration board of anatomical/medical references.
    </td>
    <td align="left" width="50%">
      Placeholder: an inspiration board of artistic and conceptual references.
    </td>
  </tr>
</table>

---

## Development story

This project grew gradually from a simple multi-planar reconstruction (MPR) viewer into a more layered environment for visual experimentation.

Some major additions included:

- multi-volume support (main + gradient + skeleton),
- split and multi-panel viewing modes,
- curved slicing planes,
- waypoint recording and playback,
- object-based modifiers in the volume space,
- heuristics and interest-scoring for slices,
- frame capture and sorting,
- and a built-in UI system that replaced earlier ImGui experiments in some versions.

### One key challenge: making the GPU and CPU viewers behave consistently

One of the biggest challenges was that the **CPU path** and the **GPU path** did not originally behave the same way.

The CPU path tended to:

1. sample the volume,
2. build panel images,
3. compose those panels together,
4. then draw the final image.

The GPU path often tried to:

1. draw slices directly to the screen,
2. and then layer UI or effects afterward.

This mismatch caused problems: some view modes worked in CPU but not GPU, side panels could appear blank, and “auto” fallback behavior became fragile.

To improve that, later versions moved toward a **two-pass GPU compositor**:

- **Pass 1:** render each slice into an offscreen texture
- **Pass 2:** composite those textures into the final panel layout

That refactor made the rendering logic more consistent and easier to debug.

### A simple diagram of the problem

<table>
  <tr>
    <td align="center" width="100%">
      <img src="images/render_pipeline_diagram.png" width="800"><br>
    </td>
  </tr>
  <tr>
    <td align="left" width="100%">
      Placeholder diagram: CPU compositor vs. GPU compositor. Show how slices are sampled, composed, and then displayed.
    </td>
  </tr>
</table>

#### Pipeline sketch

```text
Volume Data (main / gradient / skeleton)
        |
        v
Sampling Stage
  - single plane
  - axis planes
  - local oblique planes
  - curved plane
        |
        v
Panel Composition
  - single full screen
  - 3-up split screen
  - multi-volume comparison
  - side curve inspection panels
        |
        v
Optional FX / Object Interaction
  - mask / block / reflect / shift
  - fragment / pixel / branch-style effects
        |
        v
UI / Timeline / Heuristics / Capture
```

This challenge taught me that for interactive visualization systems, **consistency of the render pipeline** is as important as adding new features. A tool can have many features, but if the rendering architecture is unstable, those features become difficult to trust or extend.

---

## Features in `mpr_multivolume_full_fx_objects_v23_ui_curve_panels.py`

### Core viewing modes

- **Single view** — standard slice through the main volume
- **Single gray / invert variants**
- **Axis view** — three orthogonal panels
- **Local oblique view** — oblique slicing relative to the current plane
- **Multi-volume view** — compare main / gradient / skeleton side by side
- **Curved plane editor** — use curved sampling surfaces instead of only flat planes
- **Slice seed board** — collect slice views into a board-like display

### Curve and projection tools

- curved plane amplitude and radius controls
- side curve panels showing:
  - U section
  - V section
  - curved surface preview
- gizmo / corner preview
- custom plane orientations and oblique slicing

### Volume interventions

- 3D objects that can act as:
  - **maskers**
  - **blockers**
  - **reflectors**
  - **shifters**

### Image and frame FX

- frame / screen-space transformation modes
- pixel-grid-based reorganizations
- fragmenting, slicing, and stylization tools
- prototype computational effects inspired by growth, skeletonization, and material distortion

### Analysis and automation

- slice heuristics (filled area, blob count, circle-like blobs, tissue-like classification)
- interest scoring and recommended viewpoints
- timeline / waypoint recording
- playback and looping
- frame capture and export helpers

---

## Controls

Below is a simplified control guide based on the current v23 script. If you update the file later, this section may need to be edited to match.

### Keyboard

- **T** — cycle view mode
- **H** — hide/show mouse cursor
- **Y** — toggle live heuristics/blob counters
- **F1** — hide/show UI
- **F2** — hide/show top-right gizmo
- **F3** — suggest/apply interesting next view
- **F4** — start/stop 24fps capture
- **F5** — toggle blob debug
- **F6** — find blob-dense / low-interest view
- **F7 / F8** — cycle color filter mode / filter target
- **F9** — cycle display variant
- **F10** — toggle aux-from-main
- **F11** — cycle frame transform
- **F12** — toggle Brownian auto motion
- **M** — toggle heap modulation
- **X** — toggle 3Dconnexion input

### Built-in UI

The built-in UI includes several top-level panels / tabs:

- **Move / Brush**
- **Timeline**
- **Screen FX**
- **Objects**
- **Plane**
- **Heuristics**

There is also a **Hide UI** button, and when all overlays are hidden a small **Show UI** button appears in the corner.

### Typical interaction flow

1. Load your volumes.
2. Start in **Single** or **MultiVol** mode.
3. Adjust orientation and slice position.
4. Switch to **Curved** mode to inspect the volume using a curved plane.
5. Use the **Plane** tab to change curve amplitude/radius.
6. Turn on **SideViews** to inspect U/V/curved previews.
7. Record waypoints and capture images if needed.

---

## Gallery

Below are placeholder gallery sections. Put your final images into the `images/` folder and GIFs into the `gifs/` folder.

### 1. Multi-volume viewing

<table>
  <tr>
    <td align="center" width="33%"><img src="images/gallery_multivol_1.png" width="260"></td>
    <td align="center" width="33%"><img src="images/gallery_multivol_2.png" width="260"></td>
    <td align="center" width="33%"><img src="images/gallery_multivol_3.png" width="260"></td>
  </tr>
  <tr>
    <td align="left">Main / gradient / skeleton comparison.</td>
    <td align="left">Another multi-volume example.</td>
    <td align="left">A view emphasizing structural contrast.</td>
  </tr>
</table>

### 2. Curved plane exploration

<table>
  <tr>
    <td align="center" width="50%"><img src="images/gallery_curved_1.png" width="400"></td>
    <td align="center" width="50%"><img src="images/gallery_curved_2.png" width="400"></td>
  </tr>
  <tr>
    <td align="left">Curved slicing plane through the volume.</td>
    <td align="left">U/V side previews or alternative curve shapes.</td>
  </tr>
</table>

### 3. Object interventions and screen FX

<table>
  <tr>
    <td align="center" width="33%"><img src="images/gallery_objects_1.png" width="260"></td>
    <td align="center" width="33%"><img src="images/gallery_fx_1.png" width="260"></td>
    <td align="center" width="33%"><img src="images/gallery_fx_2.png" width="260"></td>
  </tr>
  <tr>
    <td align="left">3D object interaction with slice images.</td>
    <td align="left">Frame/screen FX example.</td>
    <td align="left">Another transformed slice output.</td>
  </tr>
</table>

### 4. Process / UI / timeline

<table>
  <tr>
    <td align="center" width="50%"><img src="images/gallery_ui_1.png" width="400"></td>
    <td align="center" width="50%"><img src="gifs/gallery_timeline.gif" width="400"></td>
  </tr>
  <tr>
    <td align="left">Built-in UI with plane and heuristics panels.</td>
    <td align="left">Timeline or playback interaction.</td>
  </tr>
</table>

---

## Folder structure

```text
project/
├── mpr_multivolume_full_fx_objects_v23_ui_curve_panels.py
├── images/
│   ├── hero_image.png
│   ├── inspiration_board_1.png
│   ├── inspiration_board_2.png
│   ├── render_pipeline_diagram.png
│   ├── gallery_multivol_1.png
│   ├── gallery_multivol_2.png
│   ├── gallery_multivol_3.png
│   ├── gallery_curved_1.png
│   ├── gallery_curved_2.png
│   ├── gallery_objects_1.png
│   ├── gallery_fx_1.png
│   ├── gallery_fx_2.png
│   └── gallery_ui_1.png
└── gifs/
    ├── hero_demo.gif
    └── gallery_timeline.gif
```

---

## Running the script

Example:

```bash
python mpr_multivolume_full_fx_objects_v23_ui_curve_panels.py
```

Make sure your volume paths inside the script point to the correct `.npy` data for:

- the **main color volume**,
- the **gradient/distance volume**,
- and the **skeleton volume**.

---

## What this project is trying to do

This project is not only a tool for viewing anatomy more clearly. It is also an experiment in how technical visualization can become a space for questioning the body.

The original medical visualization projects aimed to produce accurate knowledge through visibility. This project keeps that technical lineage, but it uses the same logic to ask different questions:

- What does it mean to stack the slices of a body?
- What happens when anatomy is projected like a map or cut like a butcher chart?
- How does a specimen become an image, and how does an image become an object?
- Where is the line between the scientific body, the hunted body, the consumed body, and the designed body?

In that sense, the viewer is both a **visualization system** and a **site of interpretation**.

---

## References

### Medical / anatomy / imaging

- National Library of Medicine. **Visible Human Project**.  
  https://www.nlm.nih.gov/research/visible/visible_human.html
- National Institutes of Health / NLM Open Data Portal. **Visible Human Project**.  
  https://openi.nlm.nih.gov/
- **Visible Human Project: normal anatomy | e-Anatomy**.  
  https://www.imaios.com/en/e-anatomy
- **CvhSlicer 2.0: Immersive and Interactive Visualization of Chinese Visible Human Data in XR**.
- **SofaAPAPI-Unity3D - Interactive Virtual Simulation of Ultrasound**.
- **OsiriX DICOM Viewer**.  
  https://www.osirix-viewer.com/
- **Visible Korean based on true color sectioned images for making realistic digital human, twenty years’ record: a review**. _Surgical and Radiologic Anatomy_.
- **Digital Fish Library - Species: Abudefduf troschelii (Panamic Sergeant Major)**.

### Computational / visualization / morphogenesis

- Jason Webb. **Morphogenesis Resources**.  
  https://github.com/jasonwebb/morphogenesis-resources
- Michael Cohen et al. **Stylized Video Cubes** (2002).
- **Image Stacks** (2003).
- Sidney Fels, Kenji Mase, Eric Lee. **Video Cubism** (1999).
- Alvaro Cassinelli. **KHRONOS PROJECTOR** (2004).
- Chris O'Shea. **Out of Bounds**.

### Artistic references

- Francesco Albano.  
  https://medinart.eu/works/francesco-albano/
- QAGOMA. **Looking at Patricia Piccinini’s monsters looking at us**.  
  https://www.qagoma.qld.gov.au/stories/looking-at-patricia-piccininis-monsters-looking-at-us/
- Thoughts Become Words. **Curious Affection: hybrids of Patricia Piccinini’s biotechnology art**.  
  https://thoughtsbecomewords.com/2018/07/22/curious-affection-hybrids-of-patricia-piccininis-biotechnology-art/

---
