python launch.py --config configs/stable-zero123.yaml --train --gpu 0 data.image_path=./load/images/bollywood_actress_rgba.png

# 3DGS + MVDream + LRM Initialization
python launch.py --config custom/threestudio-3dgs/configs/gaussian_splatting_mvdream.yaml  --train --gpu 0 system.prompt_processor.prompt="an astronaut riding a horse" system.geometry.geometry_convert_from="lrm:an astronaut riding a horse"

# 3DGS + MVDream + LRM Initialization + Annealing
python launch.py --config custom/threestudio-3dgs/configs/gaussian_splatting_mvdream_anneal.yaml  --train --gpu 0 system.prompt_processor.prompt="an astronaut riding a horse" system.geometry.geometry_convert_from="lrm:an astronaut riding a horse"

# 3DGS + SDS + LRM Initialization + Annealing
python launch.py --config custom/threestudio-3dgs/configs/gaussian_splatting_anneal.yaml  --train --gpu 0 system.prompt_processor.prompt="an astronaut riding a horse" system.geometry.geometry_convert_from="lrm:an astronaut riding a horse"

# 3DGS + CSD + LRM Initialization + Annealing
python launch.py --config custom/threestudio-3dgs/configs/gaussian_splatting_csd_anneal.yaml  --train --gpu 0 system.prompt_processor.prompt="an astronaut riding a horse" system.geometry.geometry_convert_from="lrm:an astronaut riding a horse"

python launch.py --config custom/threestudio-3dgs/configs/gaussian_splatting_csd_anneal.yaml  --train --gpu 0 system.prompt_processor.prompt="an attractive cute young famous actress, DSLR, 4K" system.geometry.geometry_convert_from="lrm:an attractive cute young famous actress, DSLR, 4K"

# 3DGS + SDS + LRM Initialization
python launch.py --config custom/threestudio-3dgs/configs/gaussian_splatting.yaml  --train --gpu 0 system.prompt_processor.prompt="an astronaut riding a horse" system.geometry.geometry_convert_from="lrm:an astronaut riding a horse"

# DreamFusion + NFSD
python launch.py --config configs/dreamfusion-nfsd.yaml --train --gpu 0 system.prompt_processor.prompt="a delicious hamburger"

# 3DGS + SV3D + LRM Initialization
python launch.py --config custom/threestudio-sv3d/configs/gaussian_splatting_sv3d.yaml --train --gpu 0 system.geometry.geometry_convert_from="lrm:an astronaut riding a horse"