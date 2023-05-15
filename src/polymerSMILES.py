# import libraries
from rdkit import Chem

class polymers():

    def __init__(self):
        pass
    
    def get_data():
        # AR103
        ar103 = "CC1CC2CCC3=CC=C(C=C3)C(CC(CC(C)C4=CC=C1C=C4)C5=CC=C2C=C5)CC(CC(CC(C)C6=CC=C(C[N+](C)(C)C)C=C6)C7=CC=C(C[N+](C)(C)C)C=C7)C8=CC=C(C[N+](C)(C)C)C=C8"
        ar103 = Chem.MolFromSmiles(ar103, sanitize=True)
        # AR204
        ar204 = "CC1CC2(C)CC(C)(CC(C)(CC(C)(CC(C)(C)C(=O)OCC[N+](C)(C)C)C(=O)OCC[N+](C)(C)C)C(=O)OCC[N+](C)(C)C)C(=O)OCCOC(=O)C(C)(C)CC(C)(CC(C)(C)C(=O)OCCOC1=O)C(=O)OCCOC2=O"
        ar204 = Chem.MolFromSmiles(ar204, sanitize=True)
        # CR61
        cr61 = "CC1CC2CCC3=CC=C(C=C3)C(CC(CCC4=CC=C1C=C4)C5=CC=C2C=C5)CC(CC(CC(C)C6=CC=C(C=C6)[S](=O)([O-])=O)C7=CC=C(C=C7)[S](=O)([O-])=O)C8=CC=C(C=C8)[S](=O)([O-])=O"
        cr61 = Chem.MolFromSmiles(cr61, sanitize=True)
        # NAFION
        nafion = "O=[S](=O)([O-])C(F)(F)C(F)(F)OC(F)(C(F)(F)F)C(F)(F)OC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(OC(F)(F)C(F)(OC(F)(F)C(F)(F)[S](=O)([O-])=O)C(F)(F)F)C(F)(F)C(F)(F)C(F)(F)C(F)(OC(F)(F)C(F)(OC(F)(F)C(F)(F)[S](=O)([O-])=O)C(F)(F)F)C(F)(F)F"
        nafion = Chem.MolFromSmiles(nafion, sanitize=True)
        # Polystrene pyridine pyridinium PSbP2VP-NMP BCE
        PSbNMP_BCE = "CC(CC(CC(CC(CC(CC(CC(CC(CCc1cccc[n+]1C)C2=NC=CC=C2)c3cccc[n+]3C)C4=NC=CC=C4)c5cccc[n+]5C)C6=NC=CC=C6)C7=CC=CC=C7)C8=CC=CC=C8)C9=CC=CC=C9"
        PSbNMP_BCE = Chem.MolFromSmiles(PSbNMP_BCE, sanitize=True)
        # Polystrene pyridine pyridinium PSbP2VP-NMP RCE
        PSbNMP_RCE = "CC(CC(CC(CC(CC(CC(CC(CC(CC(C)C1=NC=CC=C1)C2=CC=CC=N2)C3=CC=CC=C3)c4cccc[n+]4C)C5=CC=CC=C5)c6cccc[n+]6C)C7=CC=CC=C7)c8cccc[n+]8C)C9=NC=CC=C9"
        PSbNMP_RCE = Chem.MolFromSmiles(PSbNMP_RCE, sanitize=True)
        # Polvinyl alcohol sulfate PVAS RCE
        PVAS_RCE = "CC(CC(O)CC(O)CC(CC(O)CC(C)O[S]([O-])(=O)=O)O[S]([O-])(=O)=O)O[S]([O-])(=O)=O"
        PVAS_RCE = Chem.MolFromSmiles(PVAS_RCE, sanitize=True)
        # Polvinyl alcohol sulfate PVAS BCE
        PVAS_BCE = "CC(O)CC(O)CC(O)CC(CC(CC(C)O[S]([O-])(=O)=O)O[S]([O-])(=O)=O)O[S]([O-])(=O)=O"
        PVAS_BCE = Chem.MolFromSmiles(PVAS_BCE, sanitize = True)
        # CEM1 BCE
        CEM1 = "CC(CC(CC(CC1(C)CC2(C)CC(C)(C)C(=O)OCCCOCCOC(=O)C(C)(C)CC(C)(CC(C)(C)C(=O)OCCOCCCOC1=O)C(=O)OCCOCCCOC2=O)C(=O)NC(C)(C)C[S]([O-])(=O)=O)C(=O)NC(C)(C)C[S]([O-])(=O)=O)C(=O)NC(C)(C)C[S]([O-])(=O)=O"
        CEM1 = Chem.MolFromSmiles(CEM1, sanitize = True)
        # CEM2 BCE
        CEM2 = "CC(CC(CC1(C)CC2(C)CC3(C)CC(C)C(=O)OCCOCCCOC(=O)C(C)(C)CC(C)(CC(C)(CC(C)(C)C(=O)OCCCOCCOC1=O)C(=O)OCCCOCCOC2=O)C(=O)OCCCOCCOC3=O)C(=O)NC(C)(C)C[S]([O-])(=O)=O)C(=O)NC(C)(C)C[S]([O-])(=O)=O"
        CEM2 = Chem.MolFromSmiles(CEM2, sanitize = True)
        # CEM3 BCE
        CEM3 = "CC(CC1(C)CC2(C)CC3(C)CC4(C)CC(C)(C)C(=O)OCCCOCCOC(=O)C(C)CC(C)(CC(C)(CC(C)(CC(C)(C)C(=O)OCCOCCCOC1=O)C(=O)OCCOCCCOC2=O)C(=O)OCCOCCCOC3=O)C(=O)OCCOCCCOC4=O)C(=O)NC(C)(C)C[S]([O-])(=O)=O"
        CEM3 = Chem.MolFromSmiles(CEM3, sanitize = True)
        # XL_AMPS_PEGDA_n4_9percent_6units
        XLAPn4_9p = "CC(C)(C[S]([O-])(=O)=O)NC(=O)CC1C2C3C4CC(=O)OCCOCCOCCOCCOC(=O)CC(C(C(CC(=O)OCCOCCOCCOCCOC1=O)C(=O)OCCOCCOCCOCCOC2=O)C(=O)OCCOCCOCCOCCOC3=O)C(=O)OCCOCCOCCOCCOC4=O"
        XLAPn4_9p = Chem.MolFromSmiles(XLAPn4_9p, sanitize = True)
        # XL_AMPS_PEGDA_n4_45percent_6units
        XLAPn4_45p = "CC(C(C(C(C1C(C)C(=O)OCCOCCOCCOCCOC(=O)C(C)C(C)C(=O)OCCOCCOCCOCCOC1=O)C(=O)NC(C)(C)C[S]([O-])(=O)=O)C(=O)NC(C)(C)C[S]([O-])(=O)=O)C(=O)NC(C)(C)C[S]([O-])(=O)=O)C(=O)NC(C)(C)C[S]([O-])(=O)=O"
        XLAPn4_45p = Chem.MolFromSmiles(XLAPn4_45p, sanitize = True)

        # Code the names and structure files for the different systems
        polymers_names = [
                            'AR103',
                            'AR204',
                            'CR61',
                            'NAFION',
                            'PVAS_BCE',
                            'PVAS_RCE',
                            'PSbNMP_BCE',
                            'PSbNMP_RCE',
                            'CEM1',
                            'CEM2',
                            'CEM3',
                            'XLAPn4_9p',
                            'XLAPn4_45p'
                        ]

        polymers_smiles = [
                            ar103, ar204, cr61, nafion, PVAS_BCE, PVAS_RCE, 
                            PSbNMP_BCE, PSbNMP_RCE, CEM1, CEM2, CEM3, XLAPn4_9p, XLAPn4_45p
                            ]

        polymers_dict_ = dict(zip(polymers_names, polymers_smiles))

        polymers_dict_

        return polymers_dict_