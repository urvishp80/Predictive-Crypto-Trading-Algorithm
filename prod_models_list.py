models_full_list = {"obj8": {"min_max_consecutive_losses": [{"obj8": [("Class_0", "lgb_Obj 8 Linked_scale_pos_weight_4_85")]},
                                                            {"obj6": [("Class_8", "lgb_Obj 6 Linked_scale_pos_weight_2_65")]}],
                             "martingale_return": [{"obj5": [("Class_5", "lgb_Obj 5 Linked_max_bin_1024")]}]
                             },


                    "obj7": {
                             "trading_score": [{"obj8": [("Class_9", "lgb_Obj 8 Linked_scale_pos_weight_6")]},
                                               {"obj5": [("Class_10", "lgb_Obj 5 Linked_max_bin_768"),
                                                         ("Class_27", "lgb_Obj 5 Linked_scale_pos_weight_3_00")]}],
                             "martingale_return": [{"obj5": [("Class_1", "lgb_Obj 5 Linked_learning_rate_025_max_bin_1024"),
                                                             ("Class_10", "lgb_Obj 5 Linked_max_bin_768"),
                                                             ("Class_24", "lgb_Obj 5 Linked_scale_pos_weight_2_50")]}],
                             "min_max_consecutive_losses": [{"obj8": [("Class_0", "lgb_Obj 8 Linked_scale_pos_weight_4_85")]},
                                                            {"obj6": [("Class_8", "lgb_Obj 6 Linked_scale_pos_weight_2_65")]}]
                             },



                    "obj6": {"martingale_return": [{"obj6": [("Class_2", "lgb_Obj 6 Linked_scale_pos_weight_2_15"),
                                                             ("Class_8", "lgb_Obj 6 Linked_scale_pos_weight_2_65")]},
                                                   {"obj8": [("Class_0", "lgb_Obj 8 Linked_scale_pos_weight_4_85"),
                                                             ("Class_4", "lgb_Obj 8 Linked_scale_pos_weight_5_30")]}],
                             "min_max_consecutive_losses": [{"obj6": [("Class_8", "lgb_Obj 6 Linked_scale_pos_weight_2_65")]},
                                                            {"obj8": [("Class_0", "lgb_Obj 8 Linked_scale_pos_weight_4_85")]}]
                             },


                    "obj5": {"binary_score": [{"obj8": [("Class_0", "lgb_Obj 8 Linked_scale_pos_weight_4_85")]}
                                              ],
                             "trading_score": [{"obj8": [("Class_6", "lgb_Obj 8 Linked_scale_pos_weight_5_85")]}
                                               ],
                             "min_max_consecutive_losses": [{"obj6": [("Class_8", "lgb_Obj 6 Linked_scale_pos_weight_2_65")]},
                                                            {"obj8": [("Class_0", "lgb_Obj 8 Linked_scale_pos_weight_4_85")]}]
                             }
                    }

binary_score_api_response = {
    "binary_score": [
        "OBJ5OBJ8C0C",
        "OBJ5OBJ8C0P"
    ],
    "martingale_return": [
        "OBJ6OBJ6C2C",
        "OBJ6OBJ6C2P",
        "OBJ6OBJ6C8C",
        "OBJ6OBJ6C8P",
        "OBJ6OBJ8C0C",
        "OBJ6OBJ8C0P",
        "OBJ6OBJ8C4C",
        "OBJ6OBJ8C4P",
        "OBJ7OBJ5C10C",
        "OBJ7OBJ5C10P",
        "OBJ7OBJ5C1C",
        "OBJ7OBJ5C1P",
        "OBJ7OBJ5C24C",
        "OBJ7OBJ5C24P",
        "OBJ8OBJ5C5C",
        "OBJ8OBJ5C5P"
    ],
    "min_max_consecutive_losses": [
        "OBJ5OBJ6C8C",
        "OBJ5OBJ6C8P",
        "OBJ5OBJ8C0C",
        "OBJ5OBJ8C0P",
        "OBJ6OBJ6C8C",
        "OBJ6OBJ6C8P",
        "OBJ6OBJ8C0C",
        "OBJ6OBJ8C0P",
        "OBJ7OBJ6C8C",
        "OBJ7OBJ6C8P",
        "OBJ7OBJ8C0C",
        "OBJ7OBJ8C0P",
        "OBJ8OBJ6C8C",
        "OBJ8OBJ6C8P",
        "OBJ8OBJ8C0C",
        "OBJ8OBJ8C0P"
    ]
}

trading_score_api_response = {
    "trading_score": [
        "OBJ5OBJ8C6C",
        "OBJ5OBJ8C6P",
        "OBJ7OBJ5C10C",
        "OBJ7OBJ5C10P",
        "OBJ7OBJ5C27C",
        "OBJ7OBJ5C27P",
        "OBJ7OBJ8C9C",
        "OBJ7OBJ8C9P"
    ]
}
# {"obj8": [("Class 4", "lgb_Obj 8 Linked_scale_pos_weight_5_30"),("Class 0", "lgb_Obj 8 Linked_scale_pos_weight_4_85")]}
# {"obj8": [("Class 0", "lgb_Obj 8 Linked_scale_pos_weight_4_85")]},
#                                                    {"obj6": [("Class 2", "lgb_Obj 6 Linked_scale_pos_weight_2_15"),
#                                                              ("Class 8", "lgb_Obj 6 Linked_scale_pos_weight_2_65")]},
# {"obj5": [("Class 15", "lgb_Obj 5 Linked_scale_pos_weight_1_95")]}
# {"obj5": [("Class 1", "lgb_Obj 5 Linked_learning_rate_025_max_bin_1024"),
# ("Class 10", "lgb_Obj 5 Linked_max_bin_768")]}
# {"obj5": [("Class 1", "lgb_Obj 5 Linked_learning_rate_025_max_bin_1024"),
#             ("Class 10", "lgb_Obj 5 Linked_max_bin_768"),
#             ("Class 5", "lgb_Obj 5 Linked_max_bin_1024")]}
# {"obj6": [("Class 2", "lgb_Obj 6 Linked_scale_pos_weight_2_15"),
# ("Class 8", "lgb_Obj 6 Linked_scale_pos_weight_2_65")]},
# ("Class 8", "lgb_Obj 8 Linked_scale_pos_weight_5_95")
# {"obj8": [("Class 0", "lgb_Obj 8 Linked_scale_pos_weight_4_85")]}
# "binary_score": [{"obj8": [("Class 0", "lgb_Obj 8 Linked_scale_pos_weight_4_85"),
#                                                         ("Class 8", "lgb_Obj 8 Linked_scale_pos_weight_5_95")]}
#                                               ],
# "binary_score": [{"obj8": [("Class 0", "lgb_Obj 8 Linked_scale_pos_weight_4_85"),
#                                                         ("Class 8", "lgb_Obj 8 Linked_scale_pos_weight_5_95")]},
#                                               ],
# "binary_score": [{"obj8": [("Class 0", "lgb_Obj 8 Linked_scale_pos_weight_4_85")]}],
# {"obj8": [("Class 0", "lgb_Obj 8 Linked_scale_pos_weight_4_85")]}
