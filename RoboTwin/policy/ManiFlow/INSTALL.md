# Build Environment for ManiFlow Policy from scratch

1.Install Vulkan

    sudo apt install libvulkan1 mesa-vulkan-drivers vulkan-tools

---

2.Create a conda env
    
    cd RoboTwin/policy/ManiFlow/scripts

You can use a conda environment YAML file to create the env:

    conda env create -f conda_environment.yaml
    conda activate maniflow

Or create a conda env manually:

    conda create -n maniflow python=3.10
    conda activate maniflow

    # Install additional dependencies
    pip install -r requirements.txt

---

3.Install PyTorch3D, check the [official install instruction](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) if encountering any errors:

    pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

---

4.Install Curobo
    cd ../.. # go back to the root folder of RoboTwin
    cd envs
    git clone https://github.com/NVlabs/curobo.git
    cd curobo
    pip install -e . --no-build-isolation
    cd ../..

---

5.⚠️ Adjust code in sapien, mplib and curobo by using the script `scripts/modify_code.sh` to do it automatically

    bash RoboTwin/policy/ManiFlow/scripts/modify_code.sh

6.Install flash attention (optional)

normally it is not necessary, but if you want to use it, please check the [official install instruction](https://github.com/Dao-AILab/flash-attention) for more details or run the following command:

    MAX_JOBS=4 python -m pip -v install flash-attn --no-build-isolation

---

7.Install ManiFlow as a package

    cd policy/ManiFlow && pip install -e . && cd ..

----

8.install third party packages

    cd third_party
    cd gym-0.21.0 && pip install -e . && cd ..
    cd Metaworld && pip install -e . && cd ..
    cd rrl-dependencies && pip install -e mj_envs/. && pip install -e mjrl/. && cd ../
    cd r3m && pip install -e . && cd ../..

