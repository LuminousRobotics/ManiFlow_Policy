#!/bin/bash

# Update your path here
cd /home/geyan/repos/dev/ManiFlow_Policy/ # change to your ManiFlow_Policy repo path

cd RoboTwin # move to RoboTwin

# Test if pytorch3d is installed correctly and compatible with gpu support
# use script/test_pytorch3d.py
echo "Testing PyTorch3D installation..."
python policy/ManiFlow/scripts/test_pytorch3d.py
if [ $? -ne 0 ]; then
    echo "PyTorch3D installation failed or is not compatible with GPU support."
    echo "Reinstalling PyTorch3D..."
    pip uninstall -y pytorch3d
    pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
    echo "Reinstallation complete."
else
    echo "PyTorch3D is installed correctly and compatible with GPU support."
fi


# Install CuRobo
echo "Installing CuRobo..."
cd envs
git clone https://github.com/NVlabs/curobo.git
cd curobo
pip install -e . --no-build-isolation
cd ../..
echo "CuRobo installed successfully!"

echo "Adjusting code in sapien/wrapper/urdf_loader.py ..."
# location of sapien, like "~/.conda/envs/RoboTwin/lib/python3.10/site-packages/sapien"
SAPIEN_LOCATION=$(pip show sapien | grep 'Location' | awk '{print $2}')/sapien
# Adjust some code in wrapper/urdf_loader.py
URDF_LOADER=$SAPIEN_LOCATION/wrapper/urdf_loader.py
# ----------- before -----------
# 667         with open(urdf_file, "r") as f:
# 668             urdf_string = f.read()
# 669 
# 670         if srdf_file is None:
# 671             srdf_file = urdf_file[:-4] + "srdf"
# 672         if os.path.isfile(srdf_file):
# 673             with open(srdf_file, "r") as f:
# 674                 self.ignore_pairs = self.parse_srdf(f.read())
# ----------- after  -----------
# 667         with open(urdf_file, "r", encoding="utf-8") as f:
# 668             urdf_string = f.read()
# 669 
# 670         if srdf_file is None:
# 671             srdf_file = urdf_file[:-4] + ".srdf"
# 672         if os.path.isfile(srdf_file):
# 673             with open(srdf_file, "r", encoding="utf-8") as f:
# 674                 self.ignore_pairs = self.parse_srdf(f.read())
sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' $URDF_LOADER

echo "Adjusting code in mplib/planner.py ..."
# location of mplib, like "~/.conda/envs/RoboTwin/lib/python3.10/site-packages/mplib"
MPLIB_LOCATION=$(pip show mplib | grep 'Location' | awk '{print $2}')/mplib

# Adjust some code in planner.py
# ----------- before -----------
# 807             if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:
# 808                 return {"status": "screw plan failed"}
# ----------- after  ----------- 
# 807             if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:
# 808                 return {"status": "screw plan failed"}
PLANNER=$MPLIB_LOCATION/planner.py
sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' $PLANNER



echo "Installing ManiFlow..."
cd policy/ManiFlow
cd ManiFlow
pip install -e .
cd ../

echo "Installing third-party dependencies..."
# Install third party packages 
cd third_party

# Install mujoco-py (optional, if you need it)
cd mujoco-py-2.1.2.14
pip install -e .
cd ..

# Install r3m
cd r3m
pip install -e .
cd ..


echo "Finished installing third-party dependencies!"
