ORGAN_CLASSES = ('holoclone', 'meroclone', 'paraclone')
organ_base_label_ids = [1, 2]
organ_novel_label_ids = [0]

ORGAN_DESC = [
    'holoclone.',
    'meroclone.',
    'paraclone.'
]

# ORGAN_DESC = [
#     'holoclone: a clone cell, large, with smooth perimeter.',
#     'meroclone: a clone cell, wrinkled, has a wrinkled perimeter.',
#     'paraclone: a clone cell, small, highly irregular, and terminal.'
# ]

# ORGAN_DESC = [
#     'holoclone: large, with smooth perimeter. Its perimeter is nearly circular.',
#     'meroclone: a wrinkled perimeter and the area of the colony is between paraclone and holoclone.',
#     'paraclone: small, highly irregular, the colony perimeter is drawn out into marked irregularities.'
# ]

COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

LVIS_CLASSES = (
    'aerosol_can', 'air_conditioner', 'airplane', 'alarm_clock', 'alcohol',
    'alligator', 'almond', 'ambulance', 'amplifier', 'anklet', 'antenna',
    'apple', 'applesauce', 'apricot', 'apron', 'aquarium',
    'arctic_(type_of_shoe)', 'armband', 'armchair', 'armoire', 'armor',
    'artichoke', 'trash_can', 'ashtray', 'asparagus', 'atomizer',
    'avocado', 'award', 'awning', 'ax', 'baboon', 'baby_buggy',
    'basketball_backboard', 'backpack', 'handbag', 'suitcase', 'bagel',
    'bagpipe', 'baguet', 'bait', 'ball', 'ballet_skirt', 'balloon',
    'bamboo', 'banana', 'Band_Aid', 'bandage', 'bandanna', 'banjo',
    'banner', 'barbell', 'barge', 'barrel', 'barrette', 'barrow',
    'baseball_base', 'baseball', 'baseball_bat', 'baseball_cap',
    'baseball_glove', 'basket', 'basketball', 'bass_horn', 'bat_(animal)',
    'bath_mat', 'bath_towel', 'bathrobe', 'bathtub', 'batter_(food)',
    'battery', 'beachball', 'bead', 'bean_curd', 'beanbag', 'beanie',
    'bear', 'bed', 'bedpan', 'bedspread', 'cow', 'beef_(food)', 'beeper',
    'beer_bottle', 'beer_can', 'beetle', 'bell', 'bell_pepper', 'belt',
    'belt_buckle', 'bench', 'beret', 'bib', 'Bible', 'bicycle', 'visor',
    'billboard', 'binder', 'binoculars', 'bird', 'birdfeeder', 'birdbath',
    'birdcage', 'birdhouse', 'birthday_cake', 'birthday_card',
    'pirate_flag', 'black_sheep', 'blackberry', 'blackboard', 'blanket',
    'blazer', 'blender', 'blimp', 'blinker', 'blouse', 'blueberry',
    'gameboard', 'boat', 'bob', 'bobbin', 'bobby_pin', 'boiled_egg',
    'bolo_tie', 'deadbolt', 'bolt', 'bonnet', 'book', 'bookcase',
    'booklet', 'bookmark', 'boom_microphone', 'boot', 'bottle',
    'bottle_opener', 'bouquet', 'bow_(weapon)', 'bow_(decorative_ribbons)',
    'bow-tie', 'bowl', 'pipe_bowl', 'bowler_hat', 'bowling_ball', 'box',
    'boxing_glove', 'suspenders', 'bracelet', 'brass_plaque', 'brassiere',
    'bread-bin', 'bread', 'breechcloth', 'bridal_gown', 'briefcase',
    'broccoli', 'broach', 'broom', 'brownie', 'brussels_sprouts',
    'bubble_gum', 'bucket', 'horse_buggy', 'bull', 'bulldog', 'bulldozer',
    'bullet_train', 'bulletin_board', 'bulletproof_vest', 'bullhorn',
    'bun', 'bunk_bed', 'buoy', 'burrito', 'bus_(vehicle)', 'business_card',
    'butter', 'butterfly', 'button', 'cab_(taxi)', 'cabana', 'cabin_car',
    'cabinet', 'locker', 'cake', 'calculator', 'calendar', 'calf',
    'camcorder', 'camel', 'camera', 'camera_lens', 'camper_(vehicle)',
    'can', 'can_opener', 'candle', 'candle_holder', 'candy_bar',
    'candy_cane', 'walking_cane', 'canister', 'canoe', 'cantaloup',
    'canteen', 'cap_(headwear)', 'bottle_cap', 'cape', 'cappuccino',
    'car_(automobile)', 'railcar_(part_of_a_train)', 'elevator_car',
    'car_battery', 'identity_card', 'card', 'cardigan', 'cargo_ship',
    'carnation', 'horse_carriage', 'carrot', 'tote_bag', 'cart', 'carton',
    'cash_register', 'casserole', 'cassette', 'cast', 'cat', 'cauliflower',
    'cayenne_(spice)', 'CD_player', 'celery', 'cellular_telephone',
    'chain_mail', 'chair', 'chaise_longue', 'chalice', 'chandelier',
    'chap', 'checkbook', 'checkerboard', 'cherry', 'chessboard',
    'chicken_(animal)', 'chickpea', 'chili_(vegetable)', 'chime',
    'chinaware', 'crisp_(potato_chip)', 'poker_chip', 'chocolate_bar',
    'chocolate_cake', 'chocolate_milk', 'chocolate_mousse', 'choker',
    'chopping_board', 'chopstick', 'Christmas_tree', 'slide', 'cider',
    'cigar_box', 'cigarette', 'cigarette_case', 'cistern', 'clarinet',
    'clasp', 'cleansing_agent', 'cleat_(for_securing_rope)', 'clementine',
    'clip', 'clipboard', 'clippers_(for_plants)', 'cloak', 'clock',
    'clock_tower', 'clothes_hamper', 'clothespin', 'clutch_bag', 'coaster',
    'coat', 'coat_hanger', 'coatrack', 'cock', 'cockroach',
    'cocoa_(beverage)', 'coconut', 'coffee_maker', 'coffee_table',
    'coffeepot', 'coil', 'coin', 'colander', 'coleslaw',
    'coloring_material', 'combination_lock', 'pacifier', 'comic_book',
    'compass', 'computer_keyboard', 'condiment', 'cone', 'control',
    'convertible_(automobile)', 'sofa_bed', 'cooker', 'cookie',
    'cooking_utensil', 'cooler_(for_food)', 'cork_(bottle_plug)',
    'corkboard', 'corkscrew', 'edible_corn', 'cornbread', 'cornet',
    'cornice', 'cornmeal', 'corset', 'costume', 'cougar', 'coverall',
    'cowbell', 'cowboy_hat', 'crab_(animal)', 'crabmeat', 'cracker',
    'crape', 'crate', 'crayon', 'cream_pitcher', 'crescent_roll', 'crib',
    'crock_pot', 'crossbar', 'crouton', 'crow', 'crowbar', 'crown',
    'crucifix', 'cruise_ship', 'police_cruiser', 'crumb', 'crutch',
    'cub_(animal)', 'cube', 'cucumber', 'cufflink', 'cup', 'trophy_cup',
    'cupboard', 'cupcake', 'hair_curler', 'curling_iron', 'curtain',
    'cushion', 'cylinder', 'cymbal', 'dagger', 'dalmatian', 'dartboard',
    'date_(fruit)', 'deck_chair', 'deer', 'dental_floss', 'desk',
    'detergent', 'diaper', 'diary', 'die', 'dinghy', 'dining_table', 'tux',
    'dish', 'dish_antenna', 'dishrag', 'dishtowel', 'dishwasher',
    'dishwasher_detergent', 'dispenser', 'diving_board', 'Dixie_cup',
    'dog', 'dog_collar', 'doll', 'dollar', 'dollhouse', 'dolphin',
    'domestic_ass', 'doorknob', 'doormat', 'doughnut', 'dove', 'dragonfly',
    'drawer', 'underdrawers', 'dress', 'dress_hat', 'dress_suit',
    'dresser', 'drill', 'drone', 'dropper', 'drum_(musical_instrument)',
    'drumstick', 'duck', 'duckling', 'duct_tape', 'duffel_bag', 'dumbbell',
    'dumpster', 'dustpan', 'eagle', 'earphone', 'earplug', 'earring',
    'easel', 'eclair', 'eel', 'egg', 'egg_roll', 'egg_yolk', 'eggbeater',
    'eggplant', 'electric_chair', 'refrigerator', 'elephant', 'elk',
    'envelope', 'eraser', 'escargot', 'eyepatch', 'falcon', 'fan',
    'faucet', 'fedora', 'ferret', 'Ferris_wheel', 'ferry', 'fig_(fruit)',
    'fighter_jet', 'figurine', 'file_cabinet', 'file_(tool)', 'fire_alarm',
    'fire_engine', 'fire_extinguisher', 'fire_hose', 'fireplace',
    'fireplug', 'first-aid_kit', 'fish', 'fish_(food)', 'fishbowl',
    'fishing_rod', 'flag', 'flagpole', 'flamingo', 'flannel', 'flap',
    'flash', 'flashlight', 'fleece', 'flip-flop_(sandal)',
    'flipper_(footwear)', 'flower_arrangement', 'flute_glass', 'foal',
    'folding_chair', 'food_processor', 'football_(American)',
    'football_helmet', 'footstool', 'fork', 'forklift', 'freight_car',
    'French_toast', 'freshener', 'frisbee', 'frog', 'fruit_juice',
    'frying_pan', 'fudge', 'funnel', 'futon', 'gag', 'garbage',
    'garbage_truck', 'garden_hose', 'gargle', 'gargoyle', 'garlic',
    'gasmask', 'gazelle', 'gelatin', 'gemstone', 'generator',
    'giant_panda', 'gift_wrap', 'ginger', 'giraffe', 'cincture',
    'glass_(drink_container)', 'globe', 'glove', 'goat', 'goggles',
    'goldfish', 'golf_club', 'golfcart', 'gondola_(boat)', 'goose',
    'gorilla', 'gourd', 'grape', 'grater', 'gravestone', 'gravy_boat',
    'green_bean', 'green_onion', 'griddle', 'grill', 'grits', 'grizzly',
    'grocery_bag', 'guitar', 'gull', 'gun', 'hairbrush', 'hairnet',
    'hairpin', 'halter_top', 'ham', 'hamburger', 'hammer', 'hammock',
    'hamper', 'hamster', 'hair_dryer', 'hand_glass', 'hand_towel',
    'handcart', 'handcuff', 'handkerchief', 'handle', 'handsaw',
    'hardback_book', 'harmonium', 'hat', 'hatbox', 'veil', 'headband',
    'headboard', 'headlight', 'headscarf', 'headset',
    'headstall_(for_horses)', 'heart', 'heater', 'helicopter', 'helmet',
    'heron', 'highchair', 'hinge', 'hippopotamus', 'hockey_stick', 'hog',
    'home_plate_(baseball)', 'honey', 'fume_hood', 'hook', 'hookah',
    'hornet', 'horse', 'hose', 'hot-air_balloon', 'hotplate', 'hot_sauce',
    'hourglass', 'houseboat', 'hummingbird', 'hummus', 'polar_bear',
    'icecream', 'popsicle', 'ice_maker', 'ice_pack', 'ice_skate',
    'igniter', 'inhaler', 'iPod', 'iron_(for_clothing)', 'ironing_board',
    'jacket', 'jam', 'jar', 'jean', 'jeep', 'jelly_bean', 'jersey',
    'jet_plane', 'jewel', 'jewelry', 'joystick', 'jumpsuit', 'kayak',
    'keg', 'kennel', 'kettle', 'key', 'keycard', 'kilt', 'kimono',
    'kitchen_sink', 'kitchen_table', 'kite', 'kitten', 'kiwi_fruit',
    'knee_pad', 'knife', 'knitting_needle', 'knob', 'knocker_(on_a_door)',
    'koala', 'lab_coat', 'ladder', 'ladle', 'ladybug', 'lamb_(animal)',
    'lamb-chop', 'lamp', 'lamppost', 'lampshade', 'lantern', 'lanyard',
    'laptop_computer', 'lasagna', 'latch', 'lawn_mower', 'leather',
    'legging_(clothing)', 'Lego', 'legume', 'lemon', 'lemonade', 'lettuce',
    'license_plate', 'life_buoy', 'life_jacket', 'lightbulb',
    'lightning_rod', 'lime', 'limousine', 'lion', 'lip_balm', 'liquor',
    'lizard', 'log', 'lollipop', 'speaker_(stereo_equipment)', 'loveseat',
    'machine_gun', 'magazine', 'magnet', 'mail_slot', 'mailbox_(at_home)',
    'mallard', 'mallet', 'mammoth', 'manatee', 'mandarin_orange', 'manger',
    'manhole', 'map', 'marker', 'martini', 'mascot', 'mashed_potato',
    'masher', 'mask', 'mast', 'mat_(gym_equipment)', 'matchbox',
    'mattress', 'measuring_cup', 'measuring_stick', 'meatball', 'medicine',
    'melon', 'microphone', 'microscope', 'microwave_oven', 'milestone',
    'milk', 'milk_can', 'milkshake', 'minivan', 'mint_candy', 'mirror',
    'mitten', 'mixer_(kitchen_tool)', 'money',
    'monitor_(computer_equipment) computer_monitor', 'monkey', 'motor',
    'motor_scooter', 'motor_vehicle', 'motorcycle', 'mound_(baseball)',
    'mouse_(computer_equipment)', 'mousepad', 'muffin', 'mug', 'mushroom',
    'music_stool', 'musical_instrument', 'nailfile', 'napkin',
    'neckerchief', 'necklace', 'necktie', 'needle', 'nest', 'newspaper',
    'newsstand', 'nightshirt', 'nosebag_(for_animals)',
    'noseband_(for_animals)', 'notebook', 'notepad', 'nut', 'nutcracker',
    'oar', 'octopus_(food)', 'octopus_(animal)', 'oil_lamp', 'olive_oil',
    'omelet', 'onion', 'orange_(fruit)', 'orange_juice', 'ostrich',
    'ottoman', 'oven', 'overalls_(clothing)', 'owl', 'packet', 'inkpad',
    'pad', 'paddle', 'padlock', 'paintbrush', 'painting', 'pajamas',
    'palette', 'pan_(for_cooking)', 'pan_(metal_container)', 'pancake',
    'pantyhose', 'papaya', 'paper_plate', 'paper_towel', 'paperback_book',
    'paperweight', 'parachute', 'parakeet', 'parasail_(sports)', 'parasol',
    'parchment', 'parka', 'parking_meter', 'parrot',
    'passenger_car_(part_of_a_train)', 'passenger_ship', 'passport',
    'pastry', 'patty_(food)', 'pea_(food)', 'peach', 'peanut_butter',
    'pear', 'peeler_(tool_for_fruit_and_vegetables)', 'wooden_leg',
    'pegboard', 'pelican', 'pen', 'pencil', 'pencil_box',
    'pencil_sharpener', 'pendulum', 'penguin', 'pennant', 'penny_(coin)',
    'pepper', 'pepper_mill', 'perfume', 'persimmon', 'person', 'pet',
    'pew_(church_bench)', 'phonebook', 'phonograph_record', 'piano',
    'pickle', 'pickup_truck', 'pie', 'pigeon', 'piggy_bank', 'pillow',
    'pin_(non_jewelry)', 'pineapple', 'pinecone', 'ping-pong_ball',
    'pinwheel', 'tobacco_pipe', 'pipe', 'pistol', 'pita_(bread)',
    'pitcher_(vessel_for_liquid)', 'pitchfork', 'pizza', 'place_mat',
    'plate', 'platter', 'playpen', 'pliers', 'plow_(farm_equipment)',
    'plume', 'pocket_watch', 'pocketknife', 'poker_(fire_stirring_tool)',
    'pole', 'polo_shirt', 'poncho', 'pony', 'pool_table', 'pop_(soda)',
    'postbox_(public)', 'postcard', 'poster', 'pot', 'flowerpot', 'potato',
    'potholder', 'pottery', 'pouch', 'power_shovel', 'prawn', 'pretzel',
    'printer', 'projectile_(weapon)', 'projector', 'propeller', 'prune',
    'pudding', 'puffer_(fish)', 'puffin', 'pug-dog', 'pumpkin', 'puncher',
    'puppet', 'puppy', 'quesadilla', 'quiche', 'quilt', 'rabbit',
    'race_car', 'racket', 'radar', 'radiator', 'radio_receiver', 'radish',
    'raft', 'rag_doll', 'raincoat', 'ram_(animal)', 'raspberry', 'rat',
    'razorblade', 'reamer_(juicer)', 'rearview_mirror', 'receipt',
    'recliner', 'record_player', 'reflector', 'remote_control',
    'rhinoceros', 'rib_(food)', 'rifle', 'ring', 'river_boat', 'road_map',
    'robe', 'rocking_chair', 'rodent', 'roller_skate', 'Rollerblade',
    'rolling_pin', 'root_beer', 'router_(computer_equipment)',
    'rubber_band', 'runner_(carpet)', 'plastic_bag',
    'saddle_(on_an_animal)', 'saddle_blanket', 'saddlebag', 'safety_pin',
    'sail', 'salad', 'salad_plate', 'salami', 'salmon_(fish)',
    'salmon_(food)', 'salsa', 'saltshaker', 'sandal_(type_of_shoe)',
    'sandwich', 'satchel', 'saucepan', 'saucer', 'sausage', 'sawhorse',
    'saxophone', 'scale_(measuring_instrument)', 'scarecrow', 'scarf',
    'school_bus', 'scissors', 'scoreboard', 'scraper', 'screwdriver',
    'scrubbing_brush', 'sculpture', 'seabird', 'seahorse', 'seaplane',
    'seashell', 'sewing_machine', 'shaker', 'shampoo', 'shark',
    'sharpener', 'Sharpie', 'shaver_(electric)', 'shaving_cream', 'shawl',
    'shears', 'sheep', 'shepherd_dog', 'sherbert', 'shield', 'shirt',
    'shoe', 'shopping_bag', 'shopping_cart', 'short_pants', 'shot_glass',
    'shoulder_bag', 'shovel', 'shower_head', 'shower_cap',
    'shower_curtain', 'shredder_(for_paper)', 'signboard', 'silo', 'sink',
    'skateboard', 'skewer', 'ski', 'ski_boot', 'ski_parka', 'ski_pole',
    'skirt', 'skullcap', 'sled', 'sleeping_bag', 'sling_(bandage)',
    'slipper_(footwear)', 'smoothie', 'snake', 'snowboard', 'snowman',
    'snowmobile', 'soap', 'soccer_ball', 'sock', 'sofa', 'softball',
    'solar_array', 'sombrero', 'soup', 'soup_bowl', 'soupspoon',
    'sour_cream', 'soya_milk', 'space_shuttle', 'sparkler_(fireworks)',
    'spatula', 'spear', 'spectacles', 'spice_rack', 'spider', 'crawfish',
    'sponge', 'spoon', 'sportswear', 'spotlight', 'squid_(food)',
    'squirrel', 'stagecoach', 'stapler_(stapling_machine)', 'starfish',
    'statue_(sculpture)', 'steak_(food)', 'steak_knife', 'steering_wheel',
    'stepladder', 'step_stool', 'stereo_(sound_system)', 'stew', 'stirrer',
    'stirrup', 'stool', 'stop_sign', 'brake_light', 'stove', 'strainer',
    'strap', 'straw_(for_drinking)', 'strawberry', 'street_sign',
    'streetlight', 'string_cheese', 'stylus', 'subwoofer', 'sugar_bowl',
    'sugarcane_(plant)', 'suit_(clothing)', 'sunflower', 'sunglasses',
    'sunhat', 'surfboard', 'sushi', 'mop', 'sweat_pants', 'sweatband',
    'sweater', 'sweatshirt', 'sweet_potato', 'swimsuit', 'sword',
    'syringe', 'Tabasco_sauce', 'table-tennis_table', 'table',
    'table_lamp', 'tablecloth', 'tachometer', 'taco', 'tag', 'taillight',
    'tambourine', 'army_tank', 'tank_(storage_vessel)',
    'tank_top_(clothing)', 'tape_(sticky_cloth_or_paper)', 'tape_measure',
    'tapestry', 'tarp', 'tartan', 'tassel', 'tea_bag', 'teacup',
    'teakettle', 'teapot', 'teddy_bear', 'telephone', 'telephone_booth',
    'telephone_pole', 'telephoto_lens', 'television_camera',
    'television_set', 'tennis_ball', 'tennis_racket', 'tequila',
    'thermometer', 'thermos_bottle', 'thermostat', 'thimble', 'thread',
    'thumbtack', 'tiara', 'tiger', 'tights_(clothing)', 'timer', 'tinfoil',
    'tinsel', 'tissue_paper', 'toast_(food)', 'toaster', 'toaster_oven',
    'toilet', 'toilet_tissue', 'tomato', 'tongs', 'toolbox', 'toothbrush',
    'toothpaste', 'toothpick', 'cover', 'tortilla', 'tow_truck', 'towel',
    'towel_rack', 'toy', 'tractor_(farm_equipment)', 'traffic_light',
    'dirt_bike', 'trailer_truck', 'train_(railroad_vehicle)', 'trampoline',
    'tray', 'trench_coat', 'triangle_(musical_instrument)', 'tricycle',
    'tripod', 'trousers', 'truck', 'truffle_(chocolate)', 'trunk', 'vat',
    'turban', 'turkey_(food)', 'turnip', 'turtle', 'turtleneck_(clothing)',
    'typewriter', 'umbrella', 'underwear', 'unicycle', 'urinal', 'urn',
    'vacuum_cleaner', 'vase', 'vending_machine', 'vent', 'vest',
    'videotape', 'vinegar', 'violin', 'vodka', 'volleyball', 'vulture',
    'waffle', 'waffle_iron', 'wagon', 'wagon_wheel', 'walking_stick',
    'wall_clock', 'wall_socket', 'wallet', 'walrus', 'wardrobe',
    'washbasin', 'automatic_washer', 'watch', 'water_bottle',
    'water_cooler', 'water_faucet', 'water_heater', 'water_jug',
    'water_gun', 'water_scooter', 'water_ski', 'water_tower',
    'watering_can', 'watermelon', 'weathervane', 'webcam', 'wedding_cake',
    'wedding_ring', 'wet_suit', 'wheel', 'wheelchair', 'whipped_cream',
    'whistle', 'wig', 'wind_chime', 'windmill', 'window_box_(for_plants)',
    'windshield_wiper', 'windsock', 'wine_bottle', 'wine_bucket',
    'wineglass', 'blinder_(for_horses)', 'wok', 'wolf', 'wooden_spoon',
    'wreath', 'wrench', 'wristband', 'wristlet', 'yacht', 'yogurt',
    'yoke_(animal_equipment)', 'zebra', 'zucchini')
VOC_CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
            'tvmonitor')
Object365_CLASSES = ('Person', 'Sneakers', 'Chair', 'Other Shoes', 'Hat', 'Car', 'Lamp', 'Glasses', 'Bottle', 'Desk', 'Cup',
        'Street Lights', 'Cabinet/shelf', 'Handbag/Satchel', 'Bracelet', 'Plate', 'Picture/Frame', 'Helmet', 'Book',
        'Gloves', 'Storage box', 'Boat', 'Leather Shoes', 'Flower', 'Bench', 'Potted Plant', 'Bowl/Basin', 'Flag',
        'Pillow', 'Boots', 'Vase', 'Microphone', 'Necklace', 'Ring', 'SUV', 'Wine Glass', 'Belt', 'Monitor/TV',
        'Backpack', 'Umbrella', 'Traffic Light', 'Speaker', 'Watch', 'Tie', 'Trash bin Can', 'Slippers', 'Bicycle',
        'Stool', 'Barrel/bucket', 'Van', 'Couch', 'Sandals', 'Basket', 'Drum', 'Pen/Pencil', 'Bus', 'Wild Bird',
        'High Heels', 'Motorcycle', 'Guitar', 'Carpet', 'Cell Phone', 'Bread', 'Camera', 'Canned', 'Truck',
        'Traffic cone', 'Cymbal', 'Lifesaver', 'Towel', 'Stuffed Toy', 'Candle', 'Sailboat', 'Laptop', 'Awning',
        'Bed', 'Faucet', 'Tent', 'Horse', 'Mirror', 'Power outlet', 'Sink', 'Apple', 'Air Conditioner', 'Knife',
        'Hockey Stick', 'Paddle', 'Pickup Truck', 'Fork', 'Traffic Sign', 'Balloon', 'Tripod', 'Dog', 'Spoon', 'Clock',
        'Pot', 'Cow', 'Cake', 'Dinning Table', 'Sheep', 'Hanger', 'Blackboard/Whiteboard', 'Napkin', 'Other Fish',
        'Orange/Tangerine', 'Toiletry', 'Keyboard', 'Tomato', 'Lantern', 'Machinery Vehicle', 'Fan',
        'Green Vegetables', 'Banana', 'Baseball Glove', 'Airplane', 'Mouse', 'Train', 'Pumpkin', 'Soccer', 'Skiboard',
        'Luggage', 'Nightstand', 'Tea pot', 'Telephone', 'Trolley', 'Head Phone', 'Sports Car', 'Stop Sign',
        'Dessert', 'Scooter', 'Stroller', 'Crane', 'Remote', 'Refrigerator', 'Oven', 'Lemon', 'Duck', 'Baseball Bat',
        'Surveillance Camera', 'Cat', 'Jug', 'Broccoli', 'Piano', 'Pizza', 'Elephant', 'Skateboard', 'Surfboard',
        'Gun', 'Skating and Skiing shoes', 'Gas stove', 'Donut', 'Bow Tie', 'Carrot', 'Toilet', 'Kite', 'Strawberry',
        'Other Balls', 'Shovel', 'Pepper', 'Computer Box', 'Toilet Paper', 'Cleaning Products', 'Chopsticks',
        'Microwave', 'Pigeon', 'Baseball', 'Cutting/chopping Board', 'Coffee Table', 'Side Table', 'Scissors',
        'Marker', 'Pie', 'Ladder', 'Snowboard', 'Cookies', 'Radiator', 'Fire Hydrant', 'Basketball', 'Zebra', 'Grape',
        'Giraffe', 'Potato', 'Sausage', 'Tricycle', 'Violin', 'Egg', 'Fire Extinguisher', 'Candy', 'Fire Truck',
        'Billiards', 'Converter', 'Bathtub', 'Wheelchair', 'Golf Club', 'Briefcase', 'Cucumber', 'Cigar/Cigarette',
        'Paint Brush', 'Pear', 'Heavy Truck', 'Hamburger', 'Extractor', 'Extension Cord', 'Tong', 'Tennis Racket',
        'Folder', 'American Football', 'earphone', 'Mask', 'Kettle', 'Tennis', 'Ship', 'Swing', 'Coffee Machine',
        'Slide', 'Carriage', 'Onion', 'Green beans', 'Projector', 'Frisbee', 'Washing Machine/Drying Machine',
        'Chicken', 'Printer', 'Watermelon', 'Saxophone', 'Tissue', 'Toothbrush', 'Ice cream', 'Hot-air balloon',
        'Cello', 'French Fries', 'Scale', 'Trophy', 'Cabbage', 'Hot dog', 'Blender', 'Peach', 'Rice', 'Wallet/Purse',
        'Volleyball', 'Deer', 'Goose', 'Tape', 'Tablet', 'Cosmetics', 'Trumpet', 'Pineapple', 'Golf Ball',
        'Ambulance', 'Parking meter', 'Mango', 'Key', 'Hurdle', 'Fishing Rod', 'Medal', 'Flute', 'Brush', 'Penguin',
        'Megaphone', 'Corn', 'Lettuce', 'Garlic', 'Swan', 'Helicopter', 'Green Onion', 'Sandwich', 'Nuts',
        'Speed Limit Sign', 'Induction Cooker', 'Broom', 'Trombone', 'Plum', 'Rickshaw', 'Goldfish', 'Kiwi fruit',
        'Router/modem', 'Poker Card', 'Toaster', 'Shrimp', 'Sushi', 'Cheese', 'Notepaper', 'Cherry', 'Pliers', 'CD',
        'Pasta', 'Hammer', 'Cue', 'Avocado', 'Hamimelon', 'Flask', 'Mushroom', 'Screwdriver', 'Soap', 'Recorder',
        'Bear', 'Eggplant', 'Board Eraser', 'Coconut', 'Tape Measure/Ruler', 'Pig', 'Showerhead', 'Globe', 'Chips',
        'Steak', 'Crosswalk Sign', 'Stapler', 'Camel', 'Formula 1', 'Pomegranate', 'Dishwasher', 'Crab',
        'Hoverboard', 'Meat ball', 'Rice Cooker', 'Tuba', 'Calculator', 'Papaya', 'Antelope', 'Parrot', 'Seal',
        'Butterfly', 'Dumbbell', 'Donkey', 'Lion', 'Urinal', 'Dolphin', 'Electric Drill', 'Hair Dryer', 'Egg tart',
        'Jellyfish', 'Treadmill', 'Lighter', 'Grapefruit', 'Game board', 'Mop', 'Radish', 'Baozi', 'Target', 'French',
        'Spring Rolls', 'Monkey', 'Rabbit', 'Pencil Case', 'Yak', 'Red Cabbage', 'Binoculars', 'Asparagus', 'Barbell',
        'Scallop', 'Noddles', 'Comb', 'Dumpling', 'Oyster', 'Table Tennis paddle', 'Cosmetics Brush/Eyeliner Pencil',
        'Chainsaw', 'Eraser', 'Lobster', 'Durian', 'Okra', 'Lipstick', 'Cosmetics Mirror', 'Curling', 'Table Tennis')
lvis_novel_label_ids = [12, 13, 16, 19, 20, 29, 30, 37, 38, 39, 41, 48, 50, 51, 62, 68, 70, 77, 81, 84, 92, 104, 105, 112,
                    116, 118, 122, 125, 129, 130, 135, 139, 141, 143, 146, 150, 154, 158, 160, 163, 166, 171, 178, 181,
                    195, 201, 208, 209, 213, 214, 221, 222, 230, 232, 233, 235, 236, 237, 239, 243, 244, 246, 249, 250,
                    256, 257, 261, 264, 265, 268, 269, 274, 280, 281, 286, 290, 291, 293, 294, 299, 300, 301, 303, 306,
                    309, 312, 315, 316, 320, 322, 325, 330, 332, 347, 348, 351, 352, 353, 354, 356, 361, 363, 364, 365,
                    367, 373, 375, 380, 381, 387, 388, 396, 397, 399, 404, 406, 409, 412, 413, 415, 419, 425, 426, 427,
                    430, 431, 434, 438, 445, 448, 455, 457, 466, 477, 478, 479, 480, 481, 485, 487, 490, 491, 502, 505,
                    507, 508, 512, 515, 517, 526, 531, 534, 537, 540, 541, 542, 544, 550, 556, 559, 560, 566, 567, 570,
                    571, 573, 574, 576, 579, 581, 582, 584, 593, 596, 598, 601, 602, 605, 609, 615, 617, 618, 619, 624,
                    631, 633, 634, 637, 639, 645, 647, 650, 656, 661, 662, 663, 664, 670, 671, 673, 677, 685, 687, 689,
                    690, 692, 701, 709, 711, 713, 721, 726, 728, 729, 732, 742, 751, 753, 754, 757, 758, 763, 768, 771,
                    777, 778, 782, 783, 784, 786, 787, 791, 795, 802, 804, 807, 808, 809, 811, 814, 819, 821, 822, 823,
                    828, 830, 848, 849, 850, 851, 852, 854, 855, 857, 858, 861, 863, 868, 872, 882, 885, 886, 889, 890,
                    891, 893, 901, 904, 907, 912, 913, 916, 917, 919, 924, 930, 936, 937, 938, 940, 941, 943, 944, 951,
                    955, 957, 968, 971, 973, 974, 982, 984, 986, 989, 990, 991, 993, 997, 1002, 1004, 1009, 1011, 1014,
                    1015, 1027, 1028, 1029, 1030, 1031, 1046, 1047, 1048, 1052, 1053, 1056, 1057, 1074,
                    1079, 1083, 1115, 1117, 1118, 1123, 1125, 1128, 1134, 1143, 1144, 1145, 1147, 1149, 1156, 1157,
                    1158, 1164, 1166, 1192]
lvis_base_label_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33, 34,
                  35, 36, 40, 42, 43, 44, 45, 46, 47, 49, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 63, 64, 65, 66, 67,
                  69, 71,
                  72, 73, 74, 75, 76, 78, 79, 80, 82, 83, 85, 86, 87, 88, 89, 90, 91, 93, 94, 95, 96, 97, 98, 99, 100,
                  101, 102, 103, 106, 107, 108, 109, 110, 111, 113, 114, 115, 117, 119, 120, 121, 123, 124, 126, 127,
                  128, 131,
                  132, 133, 134, 136, 137, 138, 140, 142, 144, 145, 147, 148, 149, 151, 152, 153, 155, 156, 157, 159,
                  161, 162, 164, 165, 167, 168, 169, 170, 172, 173, 174, 175, 176, 177, 179, 180, 182, 183, 184, 185,
                  186, 187,
                  188, 189, 190, 191, 192, 193, 194, 196, 197, 198, 199, 200, 202, 203, 204, 205, 206, 207, 210, 211,
                  212, 215, 216, 217, 218, 219, 220, 223, 224, 225, 226, 227, 228, 229, 231, 234, 238, 240, 241, 242,
                  245, 247,
                  248, 251, 252, 253, 254, 255, 258, 259, 260, 262, 263, 266, 267, 270, 271, 272, 273, 275, 276, 277,
                  278, 279, 282, 283, 284, 285, 287, 288, 289, 292, 295, 296, 297, 298, 302, 304, 305, 307, 308, 310,
                  311, 313, 314
    , 317, 318, 319, 321, 323, 324, 326, 327, 328, 329, 331, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344,
                  345, 346, 349, 350, 355, 357, 358, 359, 360, 362, 366, 368, 369, 370, 371, 372, 374, 376, 377,
                  378, 379, 382, 383, 384, 385, 386, 389, 390, 391, 392, 393, 394, 395, 398, 400, 401, 402, 403, 405,
                  407, 408, 410, 411, 414, 416, 417, 418, 420, 421, 422, 423, 424, 428, 429, 432, 433, 435, 436, 437,
                  439, 440,
                  441, 442, 443, 444, 446, 447, 449, 450, 451, 452, 453, 454, 456, 458, 459, 460, 461, 462, 463, 464,
                  465, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 482, 483, 484, 486, 488, 489, 492, 493, 494,
                  495, 496, 497,
                  498, 499, 500, 501, 503, 504, 506, 509, 510, 511, 513, 514, 516, 518, 519, 520, 521, 522, 523, 524,
                  525, 527, 528, 529, 530, 532, 533, 535, 536, 538, 539, 543, 545, 546, 547, 548, 549, 551, 552, 553,
                  554, 555,
                  557, 558, 561, 562, 563, 564, 565, 568, 569, 572, 575, 577, 578, 580, 583, 585, 586, 587, 588, 589,
                  590, 591, 592, 594, 595, 597, 599, 600, 603, 604, 606, 607, 608, 610, 611, 612, 613, 614, 616, 620,
                  621, 622
    , 623, 625, 626, 627, 628, 629, 630, 632, 635, 636, 638, 640, 641, 642, 643, 644, 646, 648, 649, 651, 652, 653, 654,
                  655, 657, 658, 659, 660, 665, 666, 667, 668, 669, 672, 674, 675, 676, 678, 679, 680, 681, 682,
                  683, 684, 686, 688, 691, 693, 694, 695, 696, 697, 698, 699, 700, 702, 703, 704, 705, 706, 707, 708,
                  710, 712, 714, 715, 716, 717, 718, 719, 720, 722, 723, 724, 725, 727, 730, 731, 733, 734, 735, 736,
                  737, 738,
                  739, 740, 741, 743, 744, 745, 746, 747, 748, 749, 750, 752, 755, 756, 759, 760, 761, 762, 764, 765,
                  766, 767, 769, 770, 772, 773, 774, 775, 776, 779, 780, 781, 785, 788, 789, 790, 792, 793, 794, 796,
                  797, 798,
                  799, 800, 801, 803, 805, 806, 810, 812, 813, 815, 816, 817, 818, 820, 824, 825, 826, 827, 829, 831,
                  832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 853, 856, 859, 860,
                  862, 864,
                  865, 866, 867, 869, 870, 871, 873, 874, 875, 876, 877, 878, 879, 880, 881, 883, 884, 887, 888, 892,
                  894, 895, 896, 897, 898, 899, 900, 902, 903, 905, 906, 908, 909, 910, 911, 914, 915, 918, 920, 921,
                  922, 923, 925
    , 926, 927, 928, 929, 931, 932, 933, 934, 935, 939, 942, 945, 946, 947, 948, 949, 950, 952, 953, 954, 956, 958, 959,
                  960, 961, 962, 963, 964, 965, 966, 967, 969, 970, 972, 975, 976, 977, 978, 979, 980, 981, 983,
                  985, 987, 988, 992, 994, 995, 996, 998, 999, 1000, 1001, 1003, 1005, 1006, 1007, 1008, 1010, 1012,
                  1013, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1032, 1033, 1034, 1035, 1036,
                  1037,
                  1038, 1039, 1040, 1041, 1042, 1043, 1044, 1045, 1049, 1050, 1051, 1054, 1055, 1058, 1059, 1060, 1061,
                  1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1070, 1071, 1072, 1073, 1075, 1076, 1077, 1078, 1080,
                  1081, 1082
    , 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1092, 1093, 1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1102,
                  1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1116, 1119, 1120, 1121,
                  1122, 1124, 1126, 1127, 1129, 1130, 1131, 1132, 1133, 1135, 1136, 1137, 1138, 1139, 1140, 1141, 1142,
                  1146, 1148, 1150, 1151, 1152, 1153, 1154, 1155, 1159, 1160, 1161, 1162, 1163, 1165, 1167, 1168, 1169,
                  1170,
                  1171, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187,
                  1188, 1189, 1190, 1191, 1193, 1194, 1195, 1196, 1197, 1198, 1199, 1200, 1201, 1202]
voc_novel_label_ids = [15, 16, 17, 18, 19]

template_list = [
    "This is a {}",
    "There is a {}",
    "a photo of a {} in the scene",
    "a photo of a small {} in the scene",
    "a photo of a medium {} in the scene",
    "a photo of a large {} in the scene",
    "a photo of a {}",
    "a photo of a small {}",
    "a photo of a medium {}",
    "a photo of a large {}",
    "This is a photo of a {}",
    "This is a photo of a small {}",
    "This is a photo of a medium {}",
    "This is a photo of a large {}",
    "There is a {} in the scene",
    "There is the {} in the scene",
    "There is one {} in the scene",
    "This is a {} in the scene",
    "This is the {} in the scene",
    "This is one {} in the scene",
    "This is one small {} in the scene",
    "This is one medium {} in the scene",
    "This is one large {} in the scene",
    "There is a small {} in the scene",
    "There is a medium {} in the scene",
    "There is a large {} in the scene",
    "There is a {} in the photo",
    "There is the {} in the photo",
    "There is one {} in the photo",
    "There is a small {} in the photo",
    "There is the small {} in the photo",
    "There is one small {} in the photo",
    "There is a medium {} in the photo",
    "There is the medium {} in the photo",
    "There is one medium {} in the photo",
    "There is a large {} in the photo",
    "There is the large {} in the photo",
    "There is one large {} in the photo",
    "There is a {} in the picture",
    "There is the {} in the picture",
    "There is one {} in the picture",
    "There is a small {} in the picture",
    "There is the small {} in the picture",
    "There is one small {} in the picture",
    "There is a medium {} in the picture",
    "There is the medium {} in the picture",
    "There is one medium {} in the picture",
    "There is a large {} in the picture",
    "There is the large {} in the picture",
    "There is one large {} in the picture",
    "This is a {} in the photo",
    "This is the {} in the photo",
    "This is one {} in the photo",
    "This is a small {} in the photo",
    "This is the small {} in the photo",
    "This is one small {} in the photo",
    "This is a medium {} in the photo",
    "This is the medium {} in the photo",
    "This is one medium {} in the photo",
    "This is a large {} in the photo",
    "This is the large {} in the photo",
    "This is one large {} in the photo",
    "This is a {} in the picture",
    "This is the {} in the picture",
    "This is one {} in the picture",
    "This is a small {} in the picture",
    "This is the small {} in the picture",
    "This is one small {} in the picture",
    "This is a medium {} in the picture",
    "This is the medium {} in the picture",
    "This is one medium {} in the picture",
    "This is a large {} in the picture",
    "This is the large {} in the picture",
    "This is one large {} in the picture",
]
template_list1 = [
    "There is {article} {category} in the scene."
"There is the {category} in the scene."
"a photo of {article} {category} in the scene."
"a photo of the {category} in the scene."
"a photo of one {category} in the scene."
"itap of {article} {category}."
"itap of my {category}."
"itap of the {category}."
"a photo of {article} {category}."
"a photo of my {category}."
"a photo of the {category}."
"a photo of one {category}."
"a photo of many {category}."
"a good photo of {article} {category}."
"a good photo of the {category}."
"a bad photo of {article} {category}."
"a bad photo of the {category}."
"a photo of a nice {category}."
"a photo of the nice {category}."
"a photo of a cool {category}."
]
coco_unseen_ids_train = [4, 5, 9, 10, 11, 12, 15, 16, 19, 20, 25, 27, 31, 32, 34, 35, 36, 38, 40, 41, 43, 52, 55, 57, 58, 60, 66, 67, 71, 76, 77, 78]
coco_unseen_ids_test = [9, 10, 11, 12, 32, 34, 35, 38, 40, 52, 58, 60, 67, 77, 78]