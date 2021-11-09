### Validation directory configuration
config_val = dict()

config_val["source_root"] = r"C:\Users\jingyli\OwnDrive\IPA\data\2021_IPA\ori"
config_val["submission_root"] = r"C:\Users\jingyli\OwnDrive\IPA\data\2021_IPA\submission"
config_val["model_root"] = r"C:\Users\jingyli\OwnDrive\IPA\data\2021_IPA\resUnet_1634736536/checkpoint.pt"
config_val["mask_root"] = r"utils/masks.dict"
config_val["log_root"] = r"../log"
config_val["tensor_board"] = True


config_val["debug"] = False

config_val["batch_size"] = 1
config_val["num_workers"] = 1