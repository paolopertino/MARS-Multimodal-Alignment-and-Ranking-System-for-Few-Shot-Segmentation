r""" FSS-1000 few-shot semantic segmentation dataset """
import os
import glob

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np

FSS_ID_LABELS_MAPPING_TEST = {0: 'bus', 1: 'hotel_slipper', 2: 'burj_al', 3: 'reflex_camera', 4: 'abes_flyingfish', 5: 'oiltank_car', 6: 'doormat', 7: 'fish_eagle', 8: 'barber_shaver', 9: 'motorbike', 10: 'feather_clothes', 11: 'wandering_albatross', 12: 'rice_cooker', 13: 'delta_wing', 14: 'fish', 15: 'nintendo_switch', 16: 'bustard', 17: 'diver', 18: 'minicooper', 19: 'cathedrale_paris', 20: 'big_ben', 21: 'combination_lock', 22: 'villa_savoye', 23: 'american_alligator', 24: 'gym_ball', 25: 'andean_condor', 26: 'leggings', 27: 'pyramid_cube', 28: 'jet_aircraft', 29: 'meatloaf', 30: 'reel', 31: 'swan', 32: 'osprey', 33: 'crt_screen', 34: 'microscope', 35: 'rubber_eraser', 36: 'arrow', 37: 'monkey', 38: 'mitten', 39: 'spiderman', 40: 'parthenon', 41: 'bat', 42: 'chess_king', 43: 'sulphur_butterfly', 44: 'quail_egg', 45: 'oriole', 46: 'iron_man', 47: 'wooden_boat', 48: 'anise', 49: 'steering_wheel', 50: 'groenendael', 51: 'dwarf_beans', 52: 'pteropus', 53: 'chalk_brush', 54: 'bloodhound', 55: 'moon', 56: 'english_foxhound', 57: 'boxing_gloves', 58: 'peregine_falcon', 59: 'pyraminx', 60: 'cicada', 61: 'screw', 62: 'shower_curtain', 63: 'tredmill', 64: 'bulb', 65: 'bell_pepper', 66: 'lemur_catta', 67: 'doughnut', 68: 'twin_tower', 69: 'astronaut', 70: 'nintendo_3ds', 71: 'fennel_bulb', 72: 'indri', 73: 'captain_america_shield', 74: 'kunai', 75: 'broom', 76: 'iphone', 77: 'earphone1', 78: 'flying_squirrel', 79: 'onion', 80: 'vinyl', 81: 'sydney_opera_house', 82: 'oyster', 83: 'harmonica', 84: 'egg', 85: 'breast_pump', 86: 'guitar', 87: 'potato_chips', 88: 'tunnel', 89: 'cuckoo', 90: 'rubick_cube', 91: 'plastic_bag', 92: 'phonograph', 93: 'net_surface_shoes', 94: 'goldfinch', 95: 'ipad', 96: 'mite_predator', 97: 'coffee_mug', 98: 'golden_plover', 99: 'f1_racing', 100: 'lapwing', 101: 'nintendo_gba', 102: 'pizza', 103: 'rally_car', 104: 'drilling_platform', 105: 'cd', 106: 'fly', 107: 'magpie_bird', 108: 'leaf_fan', 109: 'little_blue_heron', 110: 'carriage', 111: 'moist_proof_pad', 112: 'flying_snakes', 113: 'dart_target', 114: 'warehouse_tray', 115: 'nintendo_wiiu', 116: 'chiffon_cake', 117: 'bath_ball', 118: 'manatee', 119: 'cloud', 120: 'marimba', 121: 'eagle', 122: 'ruler', 123: 'soymilk_machine', 124: 'sled', 125: 'seagull', 126: 'glider_flyingfish', 127: 'doublebus', 128: 'transport_helicopter', 129: 'window_screen', 130: 'truss_bridge', 131: 'wasp', 132: 'snowman', 133: 'poached_egg', 134: 'strawberry', 135: 'spinach', 136: 'earphone2', 137: 'downy_pitch', 138: 'taj_mahal', 139: 'rocking_chair', 140: 'cablestayed_bridge', 141: 'sealion', 142: 'banana_boat', 143: 'pheasant', 144: 'stone_lion', 145: 'electronic_stove', 146: 'fox', 147: 'iguana', 148: 'rugby_ball', 149: 'hang_glider', 150: 'water_buffalo', 151: 'lotus', 152: 'paper_plane', 153: 'missile', 154: 'flamingo', 155: 'american_chamelon', 156: 'kart', 157: 'chinese_knot', 158: 'cabbage_butterfly', 159: 'key', 160: 'church', 161: 'tiltrotor', 162: 'helicopter', 163: 'french_fries', 164: 'water_heater', 165: 'snow_leopard', 166: 'goblet', 167: 'fan', 168: 'snowplow', 169: 'leafhopper', 170: 'pspgo', 171: 'black_bear', 172: 'quail', 173: 'condor', 174: 'chandelier', 175: 'hair_razor', 176: 'white_wolf', 177: 'toaster', 178: 'pidan', 179: 'pyramid', 180: 'chicken_leg', 181: 'letter_opener', 182: 'apple_icon', 183: 'porcupine', 184: 'chicken', 185: 'stingray', 186: 'warplane', 187: 'windmill', 188: 'bamboo_slip', 189: 'wig', 190: 'flying_geckos', 191: 'stonechat', 192: 'haddock', 193: 'australian_terrier', 194: 'hover_board', 195: 'siamang', 196: 'canton_tower', 197: 'santa_sledge', 198: 'arch_bridge', 199: 'curlew', 200: 'sushi', 201: 'beet_root', 202: 'accordion', 203: 'leaf_egg', 204: 'stealth_aircraft', 205: 'stork', 206: 'bucket', 207: 'hawk', 208: 'chess_queen', 209: 'ocarina', 210: 'knife', 211: 'whippet', 212: 'cantilever_bridge', 213: 'may_bug', 214: 'wagtail', 215: 'leather_shoes', 216: 'wheelchair', 217: 'shumai', 218: 'speedboat', 219: 'vacuum_cup', 220: 'chess_knight', 221: 'pumpkin_pie', 222: 'wooden_spoon', 223: 'bamboo_dragonfly', 224: 'ganeva_chair', 225: 'soap', 226: 'clearwing_flyingfish', 227: 'pencil_sharpener1', 228: 'cricket', 229: 'photocopier', 230: 'nintendo_sp', 231: 'samarra_mosque', 232: 'clam', 233: 'charge_battery', 234: 'flying_frog', 235: 'ferrari911', 236: 'polo_shirt', 237: 'echidna', 238: 'coin', 239: 'tower_pisa'}

class DatasetFSS(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize):
        self.split = split
        self.benchmark = 'fss'
        self.shot = shot

        self.base_path = os.path.join(datapath, 'FSS-1000/data')

        # Given predefined test split, load randomly generated training/val splits:
        # (reference regarding trn/val/test splits: https://github.com/HKUSTCV/FSS-1000/issues/7))
        with open(os.path.join(datapath, 'FSS-1000/splits/%s.txt' % split), 'r') as f:
            self.categories = f.read().split('\n')[:-1]
        self.categories = sorted(self.categories)

        self.class_ids = self.build_class_ids()
        self.idx_to_classname = {i: FSS_ID_LABELS_MAPPING_TEST[i-760] for i in self.class_ids}
        self.img_metadata = self.build_img_metadata()

        self.transform = transform

    def __len__(self):
        return len(self.img_metadata)

    def __getitem__(self, idx):
        query_name, support_names, class_sample = self.sample_episode(idx)
        query_img, query_mask, support_imgs, support_masks = self.load_frame(query_name, support_names)

        query_img = self.transform(query_img)
        query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()

        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])

        support_masks_tmp = []
        for smask in support_masks:
            smask = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
            support_masks_tmp.append(smask)
        support_masks = torch.stack(support_masks_tmp)

        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'query_name': query_name,

                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'support_names': support_names,

                 'class_id': torch.tensor(class_sample)}

        return batch

    def load_frame(self, query_name, support_names):
        query_img = Image.open(query_name).convert('RGB')
        support_imgs = [Image.open(name).convert('RGB') for name in support_names]

        query_id = query_name.split('/')[-1].split('.')[0]
        query_name = os.path.join(os.path.dirname(query_name), query_id) + '.png'
        support_ids = [name.split('/')[-1].split('.')[0] for name in support_names]
        support_names = [os.path.join(os.path.dirname(name), sid) + '.png' for name, sid in zip(support_names, support_ids)]

        query_mask = self.read_mask(query_name)
        support_masks = [self.read_mask(name) for name in support_names]

        return query_img, query_mask, support_imgs, support_masks

    def read_mask(self, img_name):
        mask = torch.tensor(np.array(Image.open(img_name).convert('L')))
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        return mask

    def sample_episode(self, idx):
        query_name = self.img_metadata[idx]
        class_sample = self.categories.index(query_name.split('/')[-2])
        if self.split == 'val':
            class_sample += 520
        elif self.split == 'test':
            class_sample += 760

        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(range(1, 11), 1, replace=False)[0]
            support_name = os.path.join(os.path.dirname(query_name), str(support_name)) + '.jpg'
            if query_name != support_name: support_names.append(support_name)
            if len(support_names) == self.shot: break

        return query_name, support_names, class_sample

    def build_class_ids(self):
        if self.split == 'trn':
            class_ids = range(0, 520)
        elif self.split == 'val':
            class_ids = range(520, 760)
        elif self.split == 'test':
            class_ids = range(760, 1000)
        return class_ids

    def build_img_metadata(self):
        img_metadata = []
        for cat in self.categories:
            img_paths = sorted([path for path in glob.glob('%s/*' % os.path.join(self.base_path, cat))])
            for img_path in img_paths:
                if os.path.basename(img_path).split('.')[1] == 'jpg':
                    img_metadata.append(img_path)
        return img_metadata