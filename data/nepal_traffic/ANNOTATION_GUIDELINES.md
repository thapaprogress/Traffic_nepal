# Nepal Traffic Dataset — Annotation Guidelines

This document defines the strict labeling instructions for compiling the 5,000+ Nepal Traffic dataset. Consistent labeling is critical for training a robust YOLO-World model.

---

## 1. Class Definitions & Labeling Rules

We track **15 classes** representing the specific transportation mix of Kathmandu and other cities in Nepal.

### Class 0: `motorcycle`
* **Rules**: Label all two-wheelers (scooters, standard motorcycles, dirt bikes, electric bikes).
* **Bounds**: Bounding box should enclose the entire motorcycle. If a rider is on it, the box must enclose both the motorcycle and the rider(s).

### Class 1: `car`
* **Rules**: Standard private passenger cars, taxi sedans, hatchbacks, SUVs, and standard pick-up trucks (e.g., Hilux).

### Class 2: `bus`
* **Rules**: Large commercial buses, tourism coaches, single-decker city buses (e.g., Sajha Yatayat).
* **Note**: Exclude small passenger vans or local microbuses (annotate as `microbus` instead).

### Class 3: `truck`
* **Rules**: Large cargo trucks, construction tippers, dumpster trucks, and container shipping vehicles.

### Class 4: `microbus`
* **Rules**: Local passenger vans common in Kathmandu (e.g., Toyota HiAce, Wingy vans, local passenger microbuses).
* **Precedence**: Overrides the standard `car` or `bus` label when a van is clearly used for local public transport.

### Class 5: `tempo`
* **Rules**: Three-wheeled passenger transport vehicles (e.g., classic gas-powered or electric green/white **Safa Tempos**). Very common in Kathmandu.

### Class 6: `electric rickshaw`
* **Rules**: Battery-powered open-cabin three-wheeled rickshaws (common in Terai cities and outskirts). Distinct from Safa Tempos by being smaller and having a soft top/open frame.

### Class 7: `bicycle`
* **Rules**: All non-motorized bicycles. Include standard bicycles and cargo cycles.

### Class 8: `person`
* **Rules**: Any pedestrian, standing traffic police officer, or rider/passenger on a motorcycle/bicycle.
* **Precedence**: A person sitting on a motorcycle gets a `person` box, while the overall frame gets a `motorcycle` box. If the person is a traffic police officer, also annotate them with the `traffic police` label.

### Class 9: `helmet`
* **Rules**: Bounding box strictly enclosing a helmet worn on the head of a motorcycle rider or passenger.
* **Important**: Do not label loose helmets carried on hands or handlebars.

### Class 10: `no helmet`
* **Rules**: Enclose the head/face of a motorcycle rider or passenger who is **not** wearing a helmet.
* **Purpose**: Essential for training the helmet rule violation detection module.

### Class 11: `license plate`
* **Rules**: Bounding box strictly enclosing the license plate of any vehicle.
* **Coverage**: Label both traditional handwritten red/white/black plates and new embossed high-reflectivity plates.

### Class 12: `traffic police`
* **Rules**: Any traffic police officer in active uniform.
* **Precedence**: Draw a box enclosing the officer and label it both as `person` and `traffic police`.

### Class 13: `overloaded vehicle`
* **Rules**: Any vehicle carrying luggage, goods, or passengers far exceeding its physical boundaries.
* **Examples**: 
  - Motorcycles carrying huge boxes/containers that extend past the sides.
  - Trucks stacked high with hay or bricks.
  - Public buses with passengers hanging off the door or on the roof.

### Class 14: `school bus`
* **Rules**: Yellow student transportation buses.
* **Precedence**: Draw a box enclosing the vehicle and label it as `school bus` (do not label it as a standard `bus`).

---

## 2. General Annotation Rules

1. **Tight Bounding Boxes**: Bboxes must wrap around the objects as closely as possible. Do not leave excessive background space.
2. **Handling Occlusions**: If an object is partially hidden (e.g., a car behind a bus), annotate only the visible portion if more than 30% of it is visible. If less than 30% is visible, skip annotating that object.
3. **Nested Labels (Important)**:
   - For a motorcycle rider with a helmet:
     - 1 box for `motorcycle` (covering bike + rider).
     - 1 box for `person` (covering the rider).
     - 1 box for `helmet` (covering the helmet on head).
     - 1 box for `license plate` (if visible on the rear/front).
4. **Resolution Limits**: Do not annotate extremely small or blurry vehicles in the far background (less than ~20x20 pixels) as they only add noise to training.

---

## 3. YOLO Annotation Format Reference

Each image (e.g., `frame_0001.jpg`) must have a corresponding label file (e.g., `frame_0001.txt`) containing lines in the format:
```
<class_id> <x_center> <y_center> <width> <height>
```
* Coordinates are normalized float values (ranging from `0.0` to `1.0`) relative to the image width and height.
* Example `frame_0001.txt`:
  ```
  0 0.450 0.620 0.120 0.240   # motorcycle
  8 0.450 0.520 0.080 0.160   # person (rider)
  9 0.450 0.430 0.040 0.050   # helmet
  ```
