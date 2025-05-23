# poas-scripts

This repository contains all of the work towards the areas of interest for the Poás Volcano, including:
- IR Image Classification (`poas_ir_classifier.py`) <br> <br>

  The script can be run with: ```python<3> poas_ir_classifier.py <image_path>```, where <image_path> is the path to an IR image taken by the VPMI system at the Poas Volcano. The classifications will be displayed on the terminal. For example: <br> <br>
  ![image](https://github.com/user-attachments/assets/efca945a-80ae-435e-abc8-60c2c26f3275)

   Run: ```python3 poas_ir_classifier.py /data/vulcand/archive/imagery/infrared/345040/2024/VPMI/still/274/345040.VPMI.2024.274_041001-0000.jpg``` <br> <br>
   Output: ```Assignments for Image 345040.VPMI.2024.274_041001-0000.jpg: {'Fence', 'Plume', 'Fumaroles'}```

  <br>
- Weather <br>
  <br>
- Laguna Caliente Lake Extent (`generate_lake_extent_masks.py`)  <br> <br>

  The script can be run with: ```python3 generate_lake_extent_masks.py "year" "start_day" ("end_day")``` where `"year"` is the particular year of interest, and `<start_day>` and the optional `<end_day>` specify which days of those years to generate the lake extent binary mask for.  If no `end_day` is specified, the mask for the particular `start_day` will be generated only.  
  
  For example:  
  ```python3 generate_lake_extent_masks.py 2024 67```  
  
  Output: <br>
  ```Overlaying white areas for lake_extent/images/2024/67_lake_extent, saving to outputs/lake_extent_masks/2024/67_lake_extent.png```<br>
  ```Overlay created using a threshold of 183 votes```

  ![image](https://github.com/user-attachments/assets/3b0929af-67a5-412a-bc00-6907e9b25a9a) <br>
  <br>
- Eastern Terrace Slumping
