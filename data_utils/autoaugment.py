"""
<Reference>

We searched augmentation policy for cifar-100 dataset based on the policy search space suggested by AutoAugment paper.
The 

Autoaugmentation Policy:
[1] Philip Popien, GitHub repository, https://github.com/DeepVoltaire/AutoAugment
<AutoAugment: Learning Augmentation Policies from Data> https://arxiv.org/abs/1805.09501v1


"""

from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import random

__all__ = ['ImageNetPolicy', 'CIFAR10Policy', 'CIFAR100Policy', 'CIFAR100Policy2', 'CIFAR100Policy3', 'SVHNPolicy', 'SubPolicy']

class ImageNetPolicy():
    """ Randomly choose one of the best 24 Sub-policies on ImageNet.

        Example:
        >>> policy = ImageNetPolicy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     ImageNetPolicy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.4, "posterize", 8, 0.6, "rotate", 9, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.6, "posterize", 7, 0.6, "posterize", 6, fillcolor),
            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),

            SubPolicy(0.4, "equalize", 4, 0.8, "rotate", 8, fillcolor),
            SubPolicy(0.6, "solarize", 3, 0.6, "equalize", 7, fillcolor),
            SubPolicy(0.8, "posterize", 5, 1.0, "equalize", 2, fillcolor),
            SubPolicy(0.2, "rotate", 3, 0.6, "solarize", 8, fillcolor),
            SubPolicy(0.6, "equalize", 8, 0.4, "posterize", 6, fillcolor),

            SubPolicy(0.8, "rotate", 8, 0.4, "color", 0, fillcolor),
            SubPolicy(0.4, "rotate", 9, 0.6, "equalize", 2, fillcolor),
            SubPolicy(0.0, "equalize", 7, 0.8, "equalize", 8, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),

            SubPolicy(0.8, "rotate", 8, 1.0, "color", 2, fillcolor),
            SubPolicy(0.8, "color", 8, 0.8, "solarize", 7, fillcolor),
            SubPolicy(0.4, "sharpness", 7, 0.6, "invert", 8, fillcolor),
            SubPolicy(0.6, "shearX", 5, 1.0, "equalize", 9, fillcolor),
            SubPolicy(0.4, "color", 0, 0.6, "equalize", 3, fillcolor),

            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor)
        ]


    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment ImageNet Policy"


class CIFAR10Policy():
    """ Randomly choose one of the best 25 Sub-policies on CIFAR10.

        Example:
        >>> policy = CIFAR10Policy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     CIFAR10Policy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),
            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),
            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),

            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor),
            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),
            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),
            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),

            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),
            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),
            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),
            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),
            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),

            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),
            SubPolicy(0.2, "equalize", 8, 0.8, "equalize", 4, fillcolor),
            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),
            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor),
            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),

            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),
            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor)
        ]


    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"
    
    
class CIFAR100Policy():
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.9879223132161143, 'solarize', 1, 0.03867356636118723, 'rotate', 8, fillcolor),
            SubPolicy(0.5906539642621361, 'autocontrast', 2, 0.07767828867074882, 'shearX', 9, fillcolor),
            SubPolicy(0.18063692212392612, 'contrast', 3, 0.6762460359799197, 'translateY', 1, fillcolor),
            SubPolicy(0.4677780879889155, 'posterize', 0, 0.550614549256463, 'contrast', 2, fillcolor),
            SubPolicy(0.793757777553684, 'solarize', 4, 0.3124766163048942, 'shearY', 2, fillcolor),

            SubPolicy(0.34171542012733713, 'translateY', 8, 0.2643456363603939, 'sharpness', 5, fillcolor),
            SubPolicy(0.648144686446862, 'shearX', 5, 0.20297188174717734, 'shearY', 9, fillcolor),
            SubPolicy(0.11474619948856418, 'color', 3, 0.9131733124811067, 'autocontrast', 6, fillcolor),
            SubPolicy(0.4558313172485038, 'shearY', 0, 0.8892801924496353, 'brightness', 5, fillcolor),
            SubPolicy(0.9958622271521873, 'translateX', 1, 0.5779051595097715, 'solarize', 6, fillcolor),

            SubPolicy(0.4200533622250595, 'sharpness', 5, 0.5383153544721851, 'brightness', 4, fillcolor),
            SubPolicy(0.6409842804847393, 'shearY', 7, 0.5017245915965498, 'autocontrast', 4, fillcolor),
            SubPolicy(0.6462913003816162, 'translateX', 4, 0.27375024179538526, 'solarize', 4, fillcolor),
            SubPolicy(0.932342104383339, 'translateX', 5, 0.7433929014413744, 'sharpness', 2, fillcolor),
            SubPolicy(0.557597831322841, 'invert', 4, 0.22304480845915514, 'brightness', 4, fillcolor),

            SubPolicy(0.539726520644713, 'color', 4, 0.47647885461108325, 'equalize', 6, fillcolor),
            SubPolicy(0.6939536342980013, 'shearY', 4, 0.5242956429052371, 'color', 4, fillcolor),
            SubPolicy(0.6018510250803771, 'solarize', 1, 0.39765403790818454, 'shearY', 7, fillcolor),
            SubPolicy(0.9533833283616566, 'equalize', 8, 0.364953764116815, 'equalize', 6, fillcolor),
            SubPolicy(0.1889691085868327, 'translateX', 3, 0.5526677051817206, 'sharpness', 0, fillcolor),

            SubPolicy(0.9785734237426628, 'shearX', 4, 0.30821508771473094, 'rotate', 7, fillcolor),
            SubPolicy(0.46711128198273516, 'contrast', 8, 0.5407062800240695, 'translateY', 9, fillcolor),
            SubPolicy(0.13242120490788128, 'shearY', 9, 0.5510846119055629, 'brightness', 6, fillcolor),
            SubPolicy(0.3967473536322406, 'rotate', 3, 0.9257903359125218, 'posterize', 1, fillcolor),
            SubPolicy(0.787739760044718, 'contrast', 7, 0.06780952440551483, 'invert', 8, fillcolor)
        ]
            
    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "CIFAR100 Policy"
    
    
# This policy is the result of finding only probabiltiy
class CIFAR100Policy2():
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.10722355097540759, "invert", 7, 0.5437393796949956, "contrast", 6, fillcolor),
            SubPolicy(0.3055998483381748, "rotate", 2, 0.773173250184264, "translateX", 9, fillcolor),
            SubPolicy(0.9414919440110604, "sharpness", 1, 0.4093658906183886, "sharpness", 3, fillcolor),
            SubPolicy(0.7730489111001408, "shearY", 8, 0.26390368800969655, "translateY", 9, fillcolor),
            SubPolicy(0.6627654657594161, "autocontrast", 8, 0.18209414103505717, "equalize", 2, fillcolor),

            SubPolicy(0.9577272923915682, "shearY", 7, 0.9018599810141285, "posterize", 7, fillcolor),
            SubPolicy(0.853823844348176, "color", 3, 0.048030689963515893, "brightness", 7, fillcolor),
            SubPolicy(0.9018599810141285, "sharpness", 9, 0.9370725442876534, "brightness", 9, fillcolor),
            SubPolicy(0.7652485660086205, "equalize", 5, 0.8671833931395885, "equalize", 1, fillcolor),
            SubPolicy(0.6106424468199502, "contrast", 7, 0.6648360896925772, "sharpness", 5, fillcolor),

            SubPolicy(0.4598335650042016, "color", 7, 0.2894981099140276, "translateX", 8, fillcolor),
            SubPolicy(0.372145932516045, "equalize", 7, 0.7410543898879806, "autocontrast", 8, fillcolor),
            SubPolicy(0.6634736334238257, "translateY", 3, 0.9605265471441272, "sharpness", 6, fillcolor),
            SubPolicy(0.7005303366695717, "brightness", 6, 0.5739694427069577, "color", 8, fillcolor),
            SubPolicy(0.6238921484372798, "solarize", 2, 0.30790067383583697, "invert", 3, fillcolor),

            SubPolicy(0.6645141185305061, "equalize", 0, 0.4491491356940922, "autocontrast", 0, fillcolor),
            SubPolicy(0.5214086862517842, "equalize", 8, 0.6293260839689643, "equalize", 4, fillcolor),
            SubPolicy(0.7720261530973677, "color", 9, 0.7249478023247528, "equalize", 6, fillcolor),
            SubPolicy(0.6488103658997225, "autocontrast", 4, 0.9576658724605519, "solarize", 8, fillcolor),
            SubPolicy(0.6469099472456915, "brightness", 3, 0.8345948156359806, "color", 0, fillcolor),

            SubPolicy(0.31423772987892373, "solarize", 5, 0.7954959477085318, "autocontrast", 3, fillcolor),
            SubPolicy(0.47989872227248065, "translateY", 9, 0.35909226706871455, "translateY", 9, fillcolor),
            SubPolicy(0.4225321398670884, "autocontrast", 2, 0.15728835614960257, "solarize", 3, fillcolor),
            SubPolicy(0.6902686917033133, "equalize", 8, 0.6027040897114833, "invert", 3, fillcolor),
            SubPolicy(0.6228942701648155, "translateY", 9, 0.582029692139541, "autocontrast", 1, fillcolor)
        ]
            
    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "CIFAR100 Policy 2"
    
    
class CIFAR100Policy3():
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.578085018551005, 'posterize', 2, 0.044852206310407344, 'contrast', 5, fillcolor),
            SubPolicy(0.43106643836253644, 'contrast', 3, 0.09706135573272257, 'rotate', 7, fillcolor),
            SubPolicy(0.3582250327478384, 'rotate', 1, 0.5127269537787058, 'translateY', 1, fillcolor),
            SubPolicy(0.29124685624384844, 'contrast', 2, 0.608597934028428, 'shearY', 2, fillcolor),
            SubPolicy(0.730722055734318, 'autocontrast', 4, 0.36618924900031224, 'shearY', 0, fillcolor),

            SubPolicy(0.2878435920170448, 'translateY', 0, 0.42131158336711805, 'autocontrast', 3, fillcolor),
            SubPolicy(0.008751132832303499, 'color', 9, 0.3297029797412303, 'translateX', 5, fillcolor),
            SubPolicy(0.2709974242015076, 'solarize', 9, 0.6322071581524946, 'autocontrast', 6, fillcolor),
            SubPolicy(0.40416519975511295, 'rotate', 4, 0.6632594559009094, 'autocontrast', 6, fillcolor),
            SubPolicy(0.9944865334355615, 'shearX', 4, 0.8255270994371884, 'translateX', 6, fillcolor),

            SubPolicy(0.26653717225269746, 'translateY', 5, 0.017068249494364862, 'solarize', 8, fillcolor),
            SubPolicy(0.848403024038352, 'brightness', 9, 0.16374727558418395, 'translateY', 3, fillcolor),
            SubPolicy(0.31487073085401507, 'solarize', 4, 0.038962642979615575, 'brightness', 5, fillcolor),
            SubPolicy(0.7675079198865914, 'shearY', 7, 0.337983998863045, 'sharpness', 7, fillcolor),
            SubPolicy(0.3309630003865431, 'translateY', 9, 0.3652024405988959, 'translateX', 3, fillcolor),

            SubPolicy(0.3312809170219442, 'autocontrast', 0, 0.4888463430168321, 'autocontrast', 6, fillcolor),
            SubPolicy(0.32994588976113814, 'shearX', 3, 0.9769179967224416, 'sharpness', 1, fillcolor),
            SubPolicy(0.1340215215902173, 'autocontrast', 4, 0.23852207289137003, 'invert', 7, fillcolor),
            SubPolicy(0.6871288138441178, 'shearX', 2, 0.14988586824592662, 'sharpness', 1, fillcolor),
            SubPolicy(0.6285687923059203, 'autocontrast', 0, 0.6733023109430102, 'translateX', 0, fillcolor),

            SubPolicy(0.8619980002158236, 'translateY', 0, 0.3288516428221842, 'rotate', 1, fillcolor),
            SubPolicy(0.42522357518583864, 'brightness', 3, 0.4534317962095715, 'rotate', 3, fillcolor),
            SubPolicy(0.4051517037557008, 'shearY', 5, 0.8824125238272873, 'equalize', 3, fillcolor),
            SubPolicy(0.15491270585480324, 'equalize', 6, 0.3759600211728472, 'autocontrast', 0, fillcolor),
            SubPolicy(0.2398520204120354, 'autocontrast', 9, 0.024830059628807605, 'translateY', 6, fillcolor)
        ]
            
    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "CIFAR100 Policy 3"
    
    
class SVHNPolicy():
    """ Randomly choose one of the best 25 Sub-policies on SVHN.

        Example:
        >>> policy = SVHNPolicy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     SVHNPolicy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.9, "shearX", 4, 0.2, "invert", 3, fillcolor),
            SubPolicy(0.9, "shearY", 8, 0.7, "invert", 5, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.6, "solarize", 6, fillcolor),
            SubPolicy(0.9, "invert", 3, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.6, "equalize", 1, 0.9, "rotate", 3, fillcolor),

            SubPolicy(0.9, "shearX", 4, 0.8, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "shearY", 8, 0.4, "invert", 5, fillcolor),
            SubPolicy(0.9, "shearY", 5, 0.2, "solarize", 6, fillcolor),
            SubPolicy(0.9, "invert", 6, 0.8, "autocontrast", 1, fillcolor),
            SubPolicy(0.6, "equalize", 3, 0.9, "rotate", 3, fillcolor),

            SubPolicy(0.9, "shearX", 4, 0.3, "solarize", 3, fillcolor),
            SubPolicy(0.8, "shearY", 8, 0.7, "invert", 4, fillcolor),
            SubPolicy(0.9, "equalize", 5, 0.6, "translateY", 6, fillcolor),
            SubPolicy(0.9, "invert", 4, 0.6, "equalize", 7, fillcolor),
            SubPolicy(0.3, "contrast", 3, 0.8, "rotate", 4, fillcolor),

            SubPolicy(0.8, "invert", 5, 0.0, "translateY", 2, fillcolor),
            SubPolicy(0.7, "shearY", 6, 0.4, "solarize", 8, fillcolor),
            SubPolicy(0.6, "invert", 4, 0.8, "rotate", 4, fillcolor),
            SubPolicy(0.3, "shearY", 7, 0.9, "translateX", 3, fillcolor),
            SubPolicy(0.1, "shearX", 6, 0.6, "invert", 5, fillcolor),

            SubPolicy(0.7, "solarize", 2, 0.6, "translateY", 7, fillcolor),
            SubPolicy(0.8, "shearY", 4, 0.8, "invert", 8, fillcolor),
            SubPolicy(0.7, "shearX", 9, 0.8, "translateY", 3, fillcolor),
            SubPolicy(0.8, "shearY", 5, 0.7, "autocontrast", 3, fillcolor),
            SubPolicy(0.7, "shearX", 2, 0.1, "invert", 5, fillcolor)
        ]


    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment SVHN Policy"


class SubPolicy():
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }

        # from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)

        func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            # "rotate": lambda img, magnitude: img.rotate(magnitude * random.choice([-1, 1])),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img)
        }

        # self.name = "{}_{:.2f}_and_{}_{:.2f}".format(
        #     operation1, ranges[operation1][magnitude_idx1],
        #     operation2, ranges[operation2][magnitude_idx2])
        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]


    def __call__(self, img):
        if random.random() < self.p1: img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2: img = self.operation2(img, self.magnitude2)
        return img
