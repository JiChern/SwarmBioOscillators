



best_params_dict_gcon = { 
    
                          'roman-empire': {'nhid': 128, 'dropout': 0.3, 'e_dropout': 0.3, 'nheads': 12, 'lr': 0.01, 'nlayers': 14, 'wd': 0.0, 'Ks': 1.5, 'omega': 1.5, 'zeta': 3.0, 'dt': 0.2},

                          'Minesweeper': {'nhid': 128, 'dropout': 0.2, 'e_dropout': 0.3, 'nheads': 12, 'lr': 0.005, 'nlayers': 19, 'wd': 5e-05, 'Ks': 1.5, 'omega': 0.5, 'zeta': 2.5, 'dt': 0.4},

                          'Questions':    {'nhid': 128, 'dropout': 0.7, 'e_dropout': 0.2, 'nheads': 16, 'lr': 0.001, 'nlayers': 9, 'wd': 0.0, 'Ks': 7.5, 'omega': 3.0, 'zeta': 1.5, 'dt': 0.4},

                          'chameleon':      {'nhid': 256, 'dropout': 0.7, 'e_dropout': 0.1, 'nheads': 14, 'lr': 0.001, 'nlayers': 10, 'wd': 5e-05, 'Ks': 9.0, 'omega': 1.0, 'zeta': 2.5, 'dt': 0.3},

                          'squirrel':    {'nhid': 256, 'dropout': 0.7, 'e_dropout': 0.5, 'nheads': 14, 'lr': 0.01, 'nlayers': 5, 'wd': 5e-05, 'Ks': 1.5, 'omega': 1.5, 'zeta': 2.0, 'dt': 0.3},

                          'Amazon-ratings': {'nhid': 512, 'dropout': 0.2, 'e_dropout': 0.5, 'nheads': 1, 'lr': 0.005, 'nlayers': 4, 'wd': 0.0, 'Ks': 6.5, 'omega': 2.5, 'zeta': 1.0, 'dt': 0.2}
                          }


best_params_dict_kura = { 
                          'roman-empire': {'nhid': 128, 'dropout': 0.3, 'e_dropout': 0.3, 'nheads': 14, 'lr': 0.01, 'nlayers': 17, 'wd': 0.0, 'Ks': 0.5, 'dt': 0.1},

                          'Amazon-ratings': {'nhid': 512, 'dropout': 0.2, 'e_dropout': 0.7, 'nheads': 16, 'lr': 0.01, 'nlayers': 1, 'wd': 5e-05, 'Ks': 0.5, 'dt': 0.3},

                          'Minesweeper': {'nhid': 128, 'dropout': 0.1, 'e_dropout': 0.1, 'nheads': 12, 'lr': 0.01, 'nlayers': 1, 'wd': 0.0, 'Ks': 4.0, 'dt': 0.5},

                          'Questions':   {'nhid': 128, 'dropout': 0.1, 'e_dropout': 0.1, 'nheads': 12, 'lr': 0.001, 'nlayers': 14, 'wd': 0.0, 'Ks': 2.0, 'dt': 0.10},

                          'chameleon':   {'nhid': 512, 'dropout': 0.7, 'e_dropout': 0.2, 'nheads': 14, 'lr': 0.005, 'nlayers': 6, 'wd': 0.01, 'Ks': 3.0, 'dt': 0.1},

                          'squirrel':    {'nhid': 128, 'dropout': 0.1, 'e_dropout': 0.3, 'nheads': 12, 'lr': 0.01, 'nlayers': 3, 'wd': 0.01, 'Ks': 8.0, 'dt': 0.1},
 
                         }



best_params_dict_sies_gnn = { 
                          
                            'roman-empire': {'nhid': 128, 'dropout': 0.7, 'e_dropout': 0.7, 'nheads': 16, 'lr': 0.001, 'nlayers': 20, 'wd': 5e-05, 'dt': 0.4, 'Ks': 2.0, 'omega': 5.0, 'zeta': 4.5},

                            'Amazon-ratings': {'nhid': 256, 'dropout': 0.5, 'e_dropout': 0.3, 'nheads': 14, 'lr': 0.001, 'nlayers': 4, 'wd': 0.0, 'Ks': 1.5, 'omega': 2.5, 'zeta': 2.5, 'dt': 0.5}, 

                            'Questions': {'nhid': 128, 'dropout': 0.7, 'e_dropout': 0.5, 'nheads': 2, 'lr': 0.001, 'nlayers': 4, 'wd': 5e-05, 'Ks': 5.0, 'omega': 2.5, 'zeta': 1.0, 'dt': 0.2}, 

                            'Minesweeper':{'nhid': 64, 'dropout': 0.2, 'e_dropout': 0.2, 'nheads': 16, 'lr': 0.005, 'nlayers': 20, 'wd': 0.0, 'dt': 0.5, 'Ks': 2.0, 'omega': 5.0, 'zeta': 5.0}, 

                            'chameleon': {'nhid': 128, 'dropout': 0.7, 'e_dropout': 0.5, 'nheads': 16, 'lr': 0.001, 'nlayers': 9, 'wd': 0.0005, 'dt': 0.1, 'Ks': 8.0, 'omega': 5.0, 'zeta': 5.0},

                            'squirrel': {'nhid': 64, 'dropout': 0.7, 'e_dropout': 0.7, 'nheads': 10, 'lr': 0.001, 'nlayers': 12, 'wd': 5e-05, 'dt': 0.5, 'Ks': 11.5, 'omega': 1.5, 'zeta': 0.5} 
                            

                         }