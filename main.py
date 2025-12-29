import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Configuration graphique
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

class ACM_Cours:
    """
    ACM (Analyse des Correspondances Multiples) conforme au cours
    Toutes les variables (même quantitatives) sont transformées en qualitatives
    """
    
    def __init__(self, data, vars_quanti, vars_quali):
        """
        Paramètres:
        -----------
        data : DataFrame - données complètes
        vars_quanti : list - variables quantitatives (seront discrétisées)
        vars_quali : list - variables qualitatives
        """
        self.data = data.copy()
        self.vars_quanti_orig = vars_quanti
        self.vars_quali_orig = vars_quali
        self.n = len(data)  # n individus
        
        print("="*80)
        print("INITIALISATION DE L'ACM CONFORME AU COURS")
        print("="*80)
        print(f"Nombre d'individus (n): {self.n}")
        print(f"Variables quantitatives à discrétiser: {vars_quanti}")
        print(f"Variables qualitatives: {vars_quali}")
        print()
    
    def discretiser_variables(self, methode='quantiles', n_classes=4):
        """
        Discrétise les variables quantitatives en catégories
        """
        print("="*80)
        print("DISCRÉTISATION DES VARIABLES QUANTITATIVES")
        print("="*80)
        print(f"Méthode: {methode}")
        print(f"Nombre de classes: {n_classes}\n")
        
        self.data_discretisee = self.data.copy()
        self.info_discretisation = {}
        
        for var in self.vars_quanti_orig:
            print(f"Discrétisation de '{var}'...")
            
            try:
                if methode == 'quantiles':
                    self.data_discretisee[f'{var}_cat'], bins = pd.qcut(
                        self.data[var], 
                        q=n_classes, 
                        labels=[f'Q{i+1}' for i in range(n_classes)],
                        retbins=True,
                        duplicates='drop'
                    )
                elif methode == 'equal_width':
                    self.data_discretisee[f'{var}_cat'], bins = pd.cut(
                        self.data[var], 
                        bins=n_classes, 
                        labels=[f'C{i+1}' for i in range(n_classes)],
                        retbins=True
                    )
                
                if self.data_discretisee[f'{var}_cat'].isna().any():
                    print(f"  ⚠️  Valeurs manquantes - remplissage avec mode")
                    mode_val = self.data_discretisee[f'{var}_cat'].mode()[0]
                    self.data_discretisee[f'{var}_cat'].fillna(mode_val, inplace=True)
                    
            except Exception as e:
                print(f"  ❌ ERREUR: {e}")
                print(f"  Tentative avec 'equal_width'...")
                self.data_discretisee[f'{var}_cat'], bins = pd.cut(
                    self.data[var], 
                    bins=n_classes, 
                    labels=[f'C{i+1}' for i in range(n_classes)],
                    retbins=True
                )
            
            self.info_discretisation[var] = {
                'bins': bins,
                'n_classes': n_classes,
                'nom_categoriel': f'{var}_cat'
            }
            
            print(f"  Bornes: {[f'{b:.2f}' for b in bins]}")
            print(f"  Distribution:")
            print(self.data_discretisee[f'{var}_cat'].value_counts().sort_index())
            print()
        
        vars_discretisees = [self.info_discretisation[var]['nom_categoriel'] 
                            for var in self.vars_quanti_orig]
        self.vars_quali_toutes = self.vars_quali_orig + vars_discretisees
        
        # p = nombre de variables qualitatives
        self.p = len(self.vars_quali_toutes)
        
        print(f"✓ {len(self.vars_quanti_orig)} variables quantitatives discrétisées")
        print(f"✓ Total p = {self.p} variables qualitatives pour l'ACM")
        print()
    
    def prepare_acm(self):
        """
        Prépare les données pour l'ACM selon le cours:
        1. Codage disjonctif complet (TDC)
        2. Calcul des effectifs et fréquences
        """
        print("="*80)
        print("PRÉPARATION - TABLEAU DISJONCTIF COMPLET (TDC)")
        print("="*80)
        
        # 1. Codage disjonctif complet
        print("\n1. Construction du TDC...")
        dummies_list = []
        self.modalites_info = {}
        
        for var in self.vars_quali_toutes:
            var_dummies = pd.get_dummies(
                self.data_discretisee[var],
                prefix=var,
                prefix_sep='_'
            )
            dummies_list.append(var_dummies)
            
            modalites = self.data_discretisee[var].unique()
            self.modalites_info[var] = {
                'modalites': list(modalites),
                'm_j': len(modalites),  # nombre de modalités
                'colonnes': list(var_dummies.columns)
            }
        
        self.X = pd.concat(dummies_list, axis=1)  # Tableau disjonctif complet
        
        # 2. Calcul de J (nombre total de modalités)
        self.J = sum([self.modalites_info[var]['m_j'] for var in self.vars_quali_toutes])
        
        # 3. Effectifs des modalités (n_j)
        self.n_j = self.X.sum(axis=0)  # effectif de chaque modalité
        
        print(f"✓ TDC construit: {self.X.shape}")
        print(f"  - n = {self.n} individus")
        print(f"  - p = {self.p} variables qualitatives")
        print(f"  - J = {self.J} modalités totales")
        print(f"\n  Nombre de modalités par variable:")
        for var in self.vars_quali_toutes:
            print(f"    {var}: m_j = {self.modalites_info[var]['m_j']}")
        
        # 4. Calcul de l'inertie totale théorique
        self.I_totale_theorique = self.J / self.p - 1
        print(f"\n✓ Inertie totale théorique: I = J/p - 1 = {self.J}/{self.p} - 1 = {self.I_totale_theorique:.4f}")
        print()
    
    def compute_acm(self, n_components=5):
        """
        Calcule l'ACM via ACP du tableau disjonctif standardisé
        Selon le cours: on applique une ACP sur X avec poids 1/n pour individus
        et pondération 1/sqrt(m_j) pour variables
        """
        print("="*80)
        print("CALCUL DE L'ACM")
        print("="*80)
        
        # Nombre maximum d'axes
        self.r = min(self.n - 1, self.J - self.p)
        print(f"\nNombre maximum d'axes: r = min(n-1, J-p) = min({self.n-1}, {self.J-self.p}) = {self.r}")
        print(f"Nombre d'axes demandés: {n_components}")
        
        # Pondération des colonnes par 1/sqrt(m_j)
        X_pondere = self.X.copy()
        for var in self.vars_quali_toutes:
            m_j = self.modalites_info[var]['m_j']
            cols = self.modalites_info[var]['colonnes']
            X_pondere[cols] = X_pondere[cols] / np.sqrt(m_j)
        
        # ACP sur tableau pondéré
        n_comp = min(n_components, self.r)
        self.pca = PCA(n_components=n_comp)
        self.F = self.pca.fit_transform(X_pondere)  # Coordonnées individus F_s(i)
        
        # Valeurs propres
        self.lambda_k = self.pca.explained_variance_
        
        # Coordonnées des modalités G_s(j)
        self.G = self.pca.components_.T * np.sqrt(self.lambda_k)
        
        # Création des DataFrames pour faciliter l'utilisation
        self.coord_individus = pd.DataFrame(
            self.F,
            columns=[f'Axe{k+1}' for k in range(n_comp)],
            index=self.data.index
        )
        
        self.coord_modalites = pd.DataFrame(
            self.G,
            columns=[f'Axe{k+1}' for k in range(n_comp)],
            index=X_pondere.columns
        )
        
        # Calcul des inerties
        self._calculer_inerties()
        
        print("\n" + "="*60)
        print("VALEURS PROPRES ET INERTIES")
        print("="*60)
        print(f"{'Axe':<8} {'λ_k':<12} {'% Inertie':<12} {'% Cumulé':<12}")
        print("-"*60)
        for k in range(len(self.lambda_k)):
            print(f"Axe {k+1:<4} {self.lambda_k[k]:<12.4f} {self.inertie_pct[k]:<12.2f} {self.inertie_cum[k]:<12.2f}")
        
        print("\n" + "="*60)
        print("SEUILS IMPORTANTS (selon cours)")
        print("="*60)
        print(f"Seuil moyen: λ̄ = 1/p = 1/{self.p} = {1/self.p:.4f}")
        print(f"Maximum théorique par axe: p/(J-p) = {self.p}/{self.J-self.p} = {self.p/(self.J-self.p):.4f}")
        print(f"→ Axes à retenir: ceux avec λ_k > {1/self.p:.4f}")
        print("="*60 + "\n")
    
    def _calculer_inerties(self):
        """Calcul des différentes inerties selon le cours"""
        
        # Inertie brute (% sur inertie totale)
        self.inertie_pct = (self.lambda_k / self.I_totale_theorique) * 100
        self.inertie_cum = np.cumsum(self.inertie_pct)
        
        # Nombre d'axes significatifs (λ > 1/p)
        seuil = 1 / self.p
        self.axes_significatifs = np.where(self.lambda_k > seuil)[0]
        self.s = len(self.axes_significatifs)
        
        print(f"\nAxes significatifs (λ > 1/p = {seuil:.4f}): {self.s} axes")
        
        # Correction de Benzécri
        self.lambda_benzecri = np.array([
            ((self.p / (self.p - 1)) * (lam - 1/self.p))**2 
            if lam > 1/self.p else 0
            for lam in self.lambda_k
        ])
        
        self.S_B = np.sum(self.lambda_benzecri)
        self.inertie_benzecri = (self.lambda_benzecri / self.S_B) * 100
        self.inertie_benzecri_cum = np.cumsum(self.inertie_benzecri)
        
        # Correction de Greenacre
        somme_lambda2 = np.sum(self.lambda_k**2)
        self.S_G = (self.p / (self.p - 1)) * (somme_lambda2 - (self.J - self.p) / (self.p**2))
        self.inertie_greenacre = (self.lambda_benzecri / self.S_G) * 100
        self.inertie_greenacre_cum = np.cumsum(self.inertie_greenacre)
    
    def compute_contributions(self):
        """Calcul des contributions et cos² selon le cours"""
        print("="*80)
        print("CONTRIBUTIONS ET QUALITÉS DE REPRÉSENTATION")
        print("="*80)
        
        n_axes = len(self.lambda_k)
        
        # 1. CONTRIBUTIONS DES INDIVIDUS (formule cours: Ctr_k(i) = F_k(i)² / (n*λ_k))
        self.ctr_individus = pd.DataFrame(index=self.data.index)
        for k in range(n_axes):
            self.ctr_individus[f'Axe{k+1}'] = (self.F[:, k]**2) / (self.n * self.lambda_k[k]) * 100
        
        # 2. CONTRIBUTIONS DES MODALITÉS (formule cours: Ctr_k(j) = (n_j/np) * G_k(j)² / λ_k)
        self.ctr_modalites = pd.DataFrame(index=self.coord_modalites.index)
        for k in range(n_axes):
            self.ctr_modalites[f'Axe{k+1}'] = (
                (self.n_j / (self.n * self.p)) * (self.G[:, k]**2) / self.lambda_k[k]
            ) * 100
        
        # 3. COS² DES INDIVIDUS
        distance_ind = np.sum(self.F**2, axis=1)
        self.cos2_individus = pd.DataFrame(index=self.data.index)
        for k in range(n_axes):
            self.cos2_individus[f'Axe{k+1}'] = self.F[:, k]**2 / distance_ind
        
        # 4. COS² DES MODALITÉS
        distance_mod = np.sum(self.G**2, axis=1)
        self.cos2_modalites = pd.DataFrame(index=self.coord_modalites.index)
        for k in range(n_axes):
            self.cos2_modalites[f'Axe{k+1}'] = self.G[:, k]**2 / distance_mod
        
        # 5. CONTRIBUTIONS DES VARIABLES (somme des contributions de ses modalités)
        self.ctr_variables = pd.DataFrame(
            index=self.vars_quali_toutes,
            columns=[f'Axe{k+1}' for k in range(n_axes)]
        )
        for var in self.vars_quali_toutes:
            cols = self.modalites_info[var]['colonnes']
            for k in range(n_axes):
                self.ctr_variables.loc[var, f'Axe{k+1}'] = self.ctr_modalites.loc[cols, f'Axe{k+1}'].sum()
        
        # 6. RAPPORT DE CORRÉLATION η² (selon cours: η² = Ctr_variable × λ_k × p)
        self.eta2 = pd.DataFrame(
            index=self.vars_quali_toutes,
            columns=[f'Axe{k+1}' for k in range(n_axes)],
            dtype=float  # CORRECTION: spécifier le type float
        )
        for var in self.vars_quali_toutes:
            for k in range(n_axes):
                ctr_val = float(self.ctr_variables.loc[var, f'Axe{k+1}'])  # Conversion explicite
                self.eta2.loc[var, f'Axe{k+1}'] = (ctr_val / 100) * self.lambda_k[k] * self.p
        
        print("✓ Contributions des individus calculées")
        print("✓ Contributions des modalités calculées")
        print("✓ Contributions des variables calculées")
        print("✓ Cos² calculés")
        print("✓ Rapports de corrélation η² calculés")
        print()
    
    def afficher_tableau_inerties(self):
        """Affiche le tableau comparatif des inerties (comme dans le cours)"""
        print("\n" + "="*100)
        print("TABLEAU COMPARATIF DES INERTIES")
        print("="*100)
        
        df_inerties = pd.DataFrame({
            'λ_k': self.lambda_k,
            '% Inertie': self.inertie_pct,
            '% Cumulé': self.inertie_cum,
            'λ²_k': self.lambda_k**2,
            'λ̃_k (Benzécri)': self.lambda_benzecri,
            '% Benzécri': self.inertie_benzecri,
            '% Cum. Benz.': self.inertie_benzecri_cum,
            '% Greenacre': self.inertie_greenacre,
            '% Cum. Green.': self.inertie_greenacre_cum
        })
        df_inerties.index = [f'Axe {k+1}' for k in range(len(self.lambda_k))]
        
        print(df_inerties.to_string())
        
        print("\n" + "-"*100)
        print(f"Sommes:")
        print(f"  I_totale = {self.I_totale_theorique:.4f}")
        print(f"  S_B (Benzécri) = {self.S_B:.4f}")
        print(f"  S_G (Greenacre) = {self.S_G:.4f}")
        print("="*100 + "\n")
    
    def afficher_contributions_modalites(self, axe=1, top_n=15):
        """Affiche les modalités contribuant le plus à un axe"""
        print(f"\n{'='*80}")
        print(f"TOP {top_n} MODALITÉS CONTRIBUANT À L'AXE {axe}")
        print(f"{'='*80}")
        
        axe_col = f'Axe{axe}'
        
        # Seuil de contribution significative
        seuil_contrib = 100 / self.J  # contribution moyenne
        
        df_contrib = pd.DataFrame({
            'Coordonnée': self.coord_modalites[axe_col],
            'Contribution (%)': self.ctr_modalites[axe_col],
            'Cos²': self.cos2_modalites[axe_col],
            'Effectif': self.n_j
        })
        
        df_contrib = df_contrib.sort_values('Contribution (%)', ascending=False).head(top_n)
        
        print(df_contrib.to_string())
        print(f"\nSeuil de contribution moyenne: {seuil_contrib:.2f}%")
        print(f"Seuil |G_k(j)| > √λ_k: {np.sqrt(self.lambda_k[axe-1]):.4f}")
        print("="*80 + "\n")
    
    def afficher_eta2(self):
        """Affiche le tableau η² (rapport de corrélation)"""
        print("\n" + "="*80)
        print("RAPPORTS DE CORRÉLATION η² (Variables × Axes)")
        print("="*80)
        print("\nη² proche de 1 → Variable fortement associée à l'axe")
        print("η² proche de 0 → Variable faiblement associée à l'axe\n")
        
        print(self.eta2.to_string())
        print("\n" + "="*80 + "\n")
    
    def plot_scree(self):
        """Graphique des valeurs propres (comme dans le cours)"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        n_axes = len(self.lambda_k)
        axes_labels = [f'{k+1}' for k in range(n_axes)]
        
        # 1. Valeurs propres brutes
        ax1.bar(axes_labels, self.lambda_k, color='steelblue', 
               edgecolor='black', alpha=0.8)
        ax1.axhline(y=1/self.p, color='red', linestyle='--', linewidth=2, 
                   label=f'Seuil 1/p = {1/self.p:.3f}')
        ax1.set_xlabel('Axes', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Valeurs propres (λ_k)', fontsize=12, fontweight='bold')
        ax1.set_title('A) Valeurs Propres Brutes', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Inertie cumulée brute
        ax2.plot(axes_labels, self.inertie_cum, marker='o', 
                linewidth=2, markersize=8, color='darkgreen')
        ax2.fill_between(range(n_axes), self.inertie_cum, alpha=0.3, color='green')
        ax2.set_xlabel('Axes', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Inertie cumulée (%)', fontsize=12, fontweight='bold')
        ax2.set_title('B) Inertie Cumulée (Brute)', fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3)
        
        # 3. Comparaison Benzécri vs Greenacre
        x_pos = np.arange(n_axes)
        width = 0.35
        ax3.bar(x_pos - width/2, self.inertie_benzecri, width, 
               label='Benzécri', color='orange', edgecolor='black', alpha=0.8)
        ax3.bar(x_pos + width/2, self.inertie_greenacre, width, 
               label='Greenacre', color='purple', edgecolor='black', alpha=0.8)
        ax3.set_xlabel('Axes', fontsize=12, fontweight='bold')
        ax3.set_ylabel('% Inertie', fontsize=12, fontweight='bold')
        ax3.set_title('C) Inerties Corrigées (Benzécri vs Greenacre)', 
                     fontsize=14, fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(axes_labels)
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Inerties cumulées corrigées
        ax4.plot(axes_labels, self.inertie_benzecri_cum, marker='o', 
                linewidth=2, markersize=8, label='Benzécri', color='orange')
        ax4.plot(axes_labels, self.inertie_greenacre_cum, marker='s', 
                linewidth=2, markersize=8, label='Greenacre', color='purple')
        ax4.set_xlabel('Axes', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Inertie cumulée (%)', fontsize=12, fontweight='bold')
        ax4.set_title('D) Inerties Cumulées Corrigées', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_modalites(self, axes=(1, 2), top_contrib=None, afficher_contributions=True):
        """Projection des modalités avec contributions"""
        axe1, axe2 = axes
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        x = self.coord_modalites[f'Axe{axe1}'].values
        y = self.coord_modalites[f'Axe{axe2}'].values
        
        # Graphique 1: Coloré par cos²
        cos2_sum = self.cos2_modalites[f'Axe{axe1}'] + self.cos2_modalites[f'Axe{axe2}']
        
        ax1.axhline(y=0, color='black', linewidth=1, alpha=0.5)
        ax1.axvline(x=0, color='black', linewidth=1, alpha=0.5)
        
        scatter1 = ax1.scatter(x, y, c=cos2_sum, s=100, cmap='YlOrRd', 
                              edgecolors='black', linewidth=1, alpha=0.8)
        
        # Annoter les modalités
        if top_contrib is not None:
            contrib_axe1 = self.ctr_modalites[f'Axe{axe1}']
            contrib_axe2 = self.ctr_modalites[f'Axe{axe2}']
            contrib_tot = contrib_axe1 + contrib_axe2
            top_indices = contrib_tot.nlargest(top_contrib).index
            
            for idx in top_indices:
                pos = self.coord_modalites.index.get_loc(idx)
                ax1.annotate(idx, (x[pos], y[pos]), fontsize=8, 
                           ha='right', fontweight='bold')
        else:
            for i, label in enumerate(self.coord_modalites.index):
                ax1.annotate(label, (x[i], y[i]), fontsize=7, ha='right')
        
        plt.colorbar(scatter1, ax=ax1, label='Cos²')
        ax1.set_xlabel(f'Axe {axe1} ({self.inertie_pct[axe1-1]:.2f}%)', 
                      fontsize=12, fontweight='bold')
        ax1.set_ylabel(f'Axe {axe2} ({self.inertie_pct[axe2-1]:.2f}%)', 
                      fontsize=12, fontweight='bold')
        ax1.set_title('Modalités (colorées par Cos²)', fontsize=14, fontweight='bold')
        ax1.grid(alpha=0.3)
        
        # Graphique 2: Coloré par contribution
        if afficher_contributions:
            contrib_tot = self.ctr_modalites[f'Axe{axe1}'] + self.ctr_modalites[f'Axe{axe2}']
            
            ax2.axhline(y=0, color='black', linewidth=1, alpha=0.5)
            ax2.axvline(x=0, color='black', linewidth=1, alpha=0.5)
            
            scatter2 = ax2.scatter(x, y, c=contrib_tot, s=100, cmap='viridis', 
                                  edgecolors='black', linewidth=1, alpha=0.8)
            
            if top_contrib is not None:
                for idx in top_indices:
                    pos = self.coord_modalites.index.get_loc(idx)
                    ax2.annotate(idx, (x[pos], y[pos]), fontsize=8, 
                               ha='right', fontweight='bold')
            else:
                for i, label in enumerate(self.coord_modalites.index):
                    ax2.annotate(label, (x[i], y[i]), fontsize=7, ha='right')
            
            plt.colorbar(scatter2, ax=ax2, label='Contribution (%)')
            ax2.set_xlabel(f'Axe {axe1} ({self.inertie_pct[axe1-1]:.2f}%)', 
                          fontsize=12, fontweight='bold')
            ax2.set_ylabel(f'Axe {axe2} ({self.inertie_pct[axe2-1]:.2f}%)', 
                          fontsize=12, fontweight='bold')
            ax2.set_title('Modalités (colorées par Contribution)', 
                         fontsize=14, fontweight='bold')
            ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_individus(self, axes=(1, 2), color_by=None, top_contrib=None):
        """Projection des individus"""
        axe1, axe2 = axes
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        ax.axhline(y=0, color='black', linewidth=1, alpha=0.5)
        ax.axvline(x=0, color='black', linewidth=1, alpha=0.5)
        
        x = self.coord_individus[f'Axe{axe1}'].values
        y = self.coord_individus[f'Axe{axe2}'].values
        
        if color_by:
            if color_by in self.data.columns:
                var_color = self.data[color_by]
            elif color_by in self.data_discretisee.columns:
                var_color = self.data_discretisee[color_by]
            else:
                var_color = None
            
            if var_color is not None:
                categories = var_color.unique()
                colors = plt.cm.Set1(np.linspace(0, 1, len(categories)))
                
                for i, cat in enumerate(categories):
                    mask = var_color == cat
                    ax.scatter(x[mask], y[mask], s=50, alpha=0.6, 
                              color=colors[i], edgecolors='black', 
                              linewidth=0.5, label=str(cat))
                
                ax.legend(loc='best', fontsize=10, title=color_by)
        else:
            # Colorier par cos²
            cos2_sum = self.cos2_individus[f'Axe{axe1}'] + self.cos2_individus[f'Axe{axe2}']
            scatter = ax.scatter(x, y, s=50, alpha=0.6, c=cos2_sum, 
                               cmap='YlOrRd', edgecolors='black', linewidth=0.5)
            plt.colorbar(scatter, ax=ax, label='Cos²')
        
        # Annoter les individus avec forte contribution
        if top_contrib is not None:
            contrib_tot = self.ctr_individus[f'Axe{axe1}'] + self.ctr_individus[f'Axe{axe2}']
            top_indices = contrib_tot.nlargest(top_contrib).index
            for idx in top_indices:
                pos = self.coord_individus.index.get_loc(idx)
                ax.annotate(f'{idx}', (x[pos], y[pos]), fontsize=8, 
                           ha='right', fontweight='bold', color='red')
        
        ax.set_xlabel(f'Axe {axe1} ({self.inertie_pct[axe1-1]:.2f}%)', 
                     fontsize=12, fontweight='bold')
        ax.set_ylabel(f'Axe {axe2} ({self.inertie_pct[axe2-1]:.2f}%)', 
                     fontsize=12, fontweight='bold')
        ax.set_title('Projection des Individus', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_variables_eta2(self, axes=(1, 2)):
        """Cercle des corrélations des variables (basé sur η²)"""
        axe1, axe2 = axes
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Dessiner le cercle
        circle = plt.Circle((0, 0), 1, color='blue', fill=False, linewidth=2)
        ax.add_patch(circle)
        
        ax.axhline(y=0, color='black', linewidth=1, alpha=0.5)
        ax.axvline(x=0, color='black', linewidth=1, alpha=0.5)
        
        # Coordonnées basées sur η² - CORRECTION: conversion explicite en float
        eta2_axe1 = pd.to_numeric(self.eta2[f'Axe{axe1}'], errors='coerce').fillna(0).values
        eta2_axe2 = pd.to_numeric(self.eta2[f'Axe{axe2}'], errors='coerce').fillna(0).values
        
        x = np.sqrt(np.abs(eta2_axe1))  # abs pour éviter les racines de négatifs
        y = np.sqrt(np.abs(eta2_axe2))
        
        # Ajuster les signes selon la contribution principale
        for i, var in enumerate(self.vars_quali_toutes):
            # Vérifier le signe dominant des modalités de cette variable
            cols = self.modalites_info[var]['colonnes']
            coord_var_axe1 = self.coord_modalites.loc[cols, f'Axe{axe1}']
            coord_var_axe2 = self.coord_modalites.loc[cols, f'Axe{axe2}']
            
            if coord_var_axe1.mean() < 0:
                x[i] = -x[i]
            if coord_var_axe2.mean() < 0:
                y[i] = -y[i]
        
        # Tracer les flèches
        for i, var in enumerate(self.vars_quali_toutes):
            ax.arrow(0, 0, x[i], y[i], head_width=0.05, head_length=0.05, 
                    fc='red', ec='red', linewidth=2, alpha=0.7)
            ax.text(x[i]*1.1, y[i]*1.1, var, fontsize=11, 
                   ha='center', va='center', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_xlabel(f'Axe {axe1} ({self.inertie_pct[axe1-1]:.2f}%)', 
                     fontsize=12, fontweight='bold')
        ax.set_ylabel(f'Axe {axe2} ({self.inertie_pct[axe2-1]:.2f}%)', 
                     fontsize=12, fontweight='bold')
        ax.set_title('Représentation des Variables (basée sur η²)', 
                    fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_modalites_par_variable(self, axes=(1, 2)):
        """Projette les modalités séparées par variable (comme Figure 1.1 du cours)"""
        axe1, axe2 = axes
        
        n_vars = len(self.vars_quali_toutes)
        n_cols = 3
        n_rows = (n_vars + n_cols - 1) // n_cols
        
        fig, axes_array = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
        axes_array = axes_array.flatten() if n_vars > 1 else [axes_array]
        
        for idx, var in enumerate(self.vars_quali_toutes):
            ax = axes_array[idx]
            cols = self.modalites_info[var]['colonnes']
            
            x = self.coord_modalites.loc[cols, f'Axe{axe1}'].values
            y = self.coord_modalites.loc[cols, f'Axe{axe2}'].values
            
            ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.5)
            ax.axvline(x=0, color='black', linewidth=0.5, alpha=0.5)
            
            # Colorier par effectif
            effectifs = self.n_j[cols].values
            scatter = ax.scatter(x, y, s=200, c=effectifs, cmap='viridis', 
                               alpha=0.7, edgecolors='black', linewidth=2)
            
            for j, col in enumerate(cols):
                modalite = col.replace(f'{var}_', '')
                ax.text(x[j], y[j], modalite, fontsize=9, 
                       ha='center', va='center', fontweight='bold')
            
            plt.colorbar(scatter, ax=ax, label='Effectif')
            
            ax.set_xlabel(f'Axe {axe1} ({self.inertie_pct[axe1-1]:.2f}%)', fontsize=10)
            ax.set_ylabel(f'Axe {axe2} ({self.inertie_pct[axe2-1]:.2f}%)', fontsize=10)
            ax.set_title(f'{var}\n(m_j = {self.modalites_info[var]["m_j"]})', 
                        fontsize=12, fontweight='bold')
            ax.grid(alpha=0.3)
        
        # Cacher les axes vides
        for idx in range(n_vars, len(axes_array)):
            axes_array[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def verifier_relations_barycentriques(self, individu_id=0, modalite_id=None):
        """Vérifie les relations de transition (formules 1.15 et 1.16 du cours)"""
        print("\n" + "="*80)
        print("VÉRIFICATION DES RELATIONS BARYCENTRIQUES")
        print("="*80)
        
        # Relation 1: F_s(i) = (1/√λ_s) * (1/p) * Σ_j n_ij * G_s(j)
        print(f"\n1. Vérification pour l'individu {individu_id}:")
        print("   F_s(i) = (1/√λ_s) * (1/p) * Σ_j n_ij * G_s(j)")
        
        for s in range(min(3, len(self.lambda_k))):
            # Calcul direct
            F_direct = self.F[individu_id, s]
            
            # Calcul via relation barycentrique
            modalites_choisies = self.X.iloc[individu_id] == 1
            G_modalites = self.G[modalites_choisies, s]
            F_calcule = (1 / np.sqrt(self.lambda_k[s])) * (1 / self.p) * G_modalites.sum()
            
            print(f"   Axe {s+1}: F_direct = {F_direct:.6f}, F_calculé = {F_calcule:.6f}, "
                  f"Différence = {abs(F_direct - F_calcule):.6e}")
        
        # Relation 2: G_s(j) = (1/√λ_s) * (1/n_j) * Σ_i n_ij * F_s(i)
        if modalite_id is None:
            modalite_id = self.coord_modalites.index[0]
        
        print(f"\n2. Vérification pour la modalité '{modalite_id}':")
        print("   G_s(j) = (1/√λ_s) * (1/n_j) * Σ_i n_ij * F_s(i)")
        
        j_idx = self.coord_modalites.index.get_loc(modalite_id)
        n_j_val = self.n_j.iloc[j_idx]
        
        for s in range(min(3, len(self.lambda_k))):
            # Calcul direct
            G_direct = self.G[j_idx, s]
            
            # Calcul via relation barycentrique
            individus_avec_modalite = self.X.iloc[:, j_idx] == 1
            F_individus = self.F[individus_avec_modalite, s]
            G_calcule = (1 / np.sqrt(self.lambda_k[s])) * (1 / n_j_val) * F_individus.sum()
            
            print(f"   Axe {s+1}: G_direct = {G_direct:.6f}, G_calculé = {G_calcule:.6f}, "
                  f"Différence = {abs(G_direct - G_calcule):.6e}")
        
        print("\n✓ Les relations barycentriques sont vérifiées!")
        print("="*80 + "\n")
    
    def plot_modalites_3d(self, axes=(1, 2, 3), top_contrib=None, rotation=(30, 45)):
        """Projection 3D des modalités"""
        axe1, axe2, axe3 = axes
        
        fig = plt.figure(figsize=(16, 12))
        
        # Sous-graphique 1: Coloré par cos²
        ax1 = fig.add_subplot(121, projection='3d')
        
        x = self.coord_modalites[f'Axe{axe1}'].values
        y = self.coord_modalites[f'Axe{axe2}'].values
        z = self.coord_modalites[f'Axe{axe3}'].values
        
        cos2_sum = (self.cos2_modalites[f'Axe{axe1}'] + 
                    self.cos2_modalites[f'Axe{axe2}'] + 
                    self.cos2_modalites[f'Axe{axe3}'])
        
        scatter1 = ax1.scatter(x, y, z, c=cos2_sum, s=100, cmap='YlOrRd', 
                              edgecolors='black', linewidth=1, alpha=0.8)
        
        # Annoter les modalités
        if top_contrib is not None:
            contrib_tot = (self.ctr_modalites[f'Axe{axe1}'] + 
                          self.ctr_modalites[f'Axe{axe2}'] + 
                          self.ctr_modalites[f'Axe{axe3}'])
            top_indices = contrib_tot.nlargest(top_contrib).index
            
            for idx in top_indices:
                pos = self.coord_modalites.index.get_loc(idx)
                ax1.text(x[pos], y[pos], z[pos], idx, fontsize=7, 
                        fontweight='bold')
        else:
            for i, label in enumerate(self.coord_modalites.index):
                ax1.text(x[i], y[i], z[i], label, fontsize=6)
        
        # Plans de référence
        ax1.plot([0, 0], [0, 0], [z.min(), z.max()], 'k-', alpha=0.3, linewidth=0.5)
        ax1.plot([0, 0], [y.min(), y.max()], [0, 0], 'k-', alpha=0.3, linewidth=0.5)
        ax1.plot([x.min(), x.max()], [0, 0], [0, 0], 'k-', alpha=0.3, linewidth=0.5)
        
        ax1.set_xlabel(f'Axe {axe1} ({self.inertie_pct[axe1-1]:.2f}%)', 
                      fontsize=10, fontweight='bold')
        ax1.set_ylabel(f'Axe {axe2} ({self.inertie_pct[axe2-1]:.2f}%)', 
                      fontsize=10, fontweight='bold')
        ax1.set_zlabel(f'Axe {axe3} ({self.inertie_pct[axe3-1]:.2f}%)', 
                      fontsize=10, fontweight='bold')
        ax1.set_title('Modalités 3D (colorées par Cos²)', 
                     fontsize=12, fontweight='bold', pad=20)
        ax1.view_init(elev=rotation[0], azim=rotation[1])
        fig.colorbar(scatter1, ax=ax1, label='Cos²', shrink=0.5, pad=0.1)
        
        # Sous-graphique 2: Coloré par contribution
        ax2 = fig.add_subplot(122, projection='3d')
        
        contrib_tot = (self.ctr_modalites[f'Axe{axe1}'] + 
                      self.ctr_modalites[f'Axe{axe2}'] + 
                      self.ctr_modalites[f'Axe{axe3}'])
        
        scatter2 = ax2.scatter(x, y, z, c=contrib_tot, s=100, cmap='viridis', 
                              edgecolors='black', linewidth=1, alpha=0.8)
        
        if top_contrib is not None:
            for idx in top_indices:
                pos = self.coord_modalites.index.get_loc(idx)
                ax2.text(x[pos], y[pos], z[pos], idx, fontsize=7, 
                        fontweight='bold')
        else:
            for i, label in enumerate(self.coord_modalites.index):
                ax2.text(x[i], y[i], z[i], label, fontsize=6)
        
        # Plans de référence
        ax2.plot([0, 0], [0, 0], [z.min(), z.max()], 'k-', alpha=0.3, linewidth=0.5)
        ax2.plot([0, 0], [y.min(), y.max()], [0, 0], 'k-', alpha=0.3, linewidth=0.5)
        ax2.plot([x.min(), x.max()], [0, 0], [0, 0], 'k-', alpha=0.3, linewidth=0.5)
        
        ax2.set_xlabel(f'Axe {axe1} ({self.inertie_pct[axe1-1]:.2f}%)', 
                      fontsize=10, fontweight='bold')
        ax2.set_ylabel(f'Axe {axe2} ({self.inertie_pct[axe2-1]:.2f}%)', 
                      fontsize=10, fontweight='bold')
        ax2.set_zlabel(f'Axe {axe3} ({self.inertie_pct[axe3-1]:.2f}%)', 
                      fontsize=10, fontweight='bold')
        ax2.set_title('Modalités 3D (colorées par Contribution)', 
                     fontsize=12, fontweight='bold', pad=20)
        ax2.view_init(elev=rotation[0], azim=rotation[1])
        fig.colorbar(scatter2, ax=ax2, label='Contribution (%)', shrink=0.5, pad=0.1)
        
        plt.tight_layout()
        plt.show()
    
    def plot_individus_3d(self, axes=(1, 2, 3), color_by=None, top_contrib=None, rotation=(30, 45)):
        """Projection 3D des individus"""
        axe1, axe2, axe3 = axes
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        x = self.coord_individus[f'Axe{axe1}'].values
        y = self.coord_individus[f'Axe{axe2}'].values
        z = self.coord_individus[f'Axe{axe3}'].values
        
        if color_by:
            if color_by in self.data.columns:
                var_color = self.data[color_by]
            elif color_by in self.data_discretisee.columns:
                var_color = self.data_discretisee[color_by]
            else:
                var_color = None
            
            if var_color is not None:
                categories = var_color.unique()
                colors = plt.cm.Set1(np.linspace(0, 1, len(categories)))
                
                for i, cat in enumerate(categories):
                    mask = var_color == cat
                    ax.scatter(x[mask], y[mask], z[mask], s=50, alpha=0.6, 
                              color=colors[i], edgecolors='black', 
                              linewidth=0.5, label=str(cat))
                
                ax.legend(loc='best', fontsize=9, title=color_by)
        else:
            cos2_sum = (self.cos2_individus[f'Axe{axe1}'] + 
                       self.cos2_individus[f'Axe{axe2}'] + 
                       self.cos2_individus[f'Axe{axe3}'])
            scatter = ax.scatter(x, y, z, s=50, alpha=0.6, c=cos2_sum, 
                               cmap='YlOrRd', edgecolors='black', linewidth=0.5)
            fig.colorbar(scatter, ax=ax, label='Cos²', shrink=0.5, pad=0.1)
        
        # Annoter les individus avec forte contribution
        if top_contrib is not None:
            contrib_tot = (self.ctr_individus[f'Axe{axe1}'] + 
                          self.ctr_individus[f'Axe{axe2}'] + 
                          self.ctr_individus[f'Axe{axe3}'])
            top_indices = contrib_tot.nlargest(top_contrib).index
            for idx in top_indices:
                pos = self.coord_individus.index.get_loc(idx)
                ax.text(x[pos], y[pos], z[pos], f'{idx}', fontsize=7, 
                       fontweight='bold', color='red')
        
        # Plans de référence
        ax.plot([0, 0], [0, 0], [z.min(), z.max()], 'k-', alpha=0.3, linewidth=0.5)
        ax.plot([0, 0], [y.min(), y.max()], [0, 0], 'k-', alpha=0.3, linewidth=0.5)
        ax.plot([x.min(), x.max()], [0, 0], [0, 0], 'k-', alpha=0.3, linewidth=0.5)
        
        ax.set_xlabel(f'Axe {axe1} ({self.inertie_pct[axe1-1]:.2f}%)', 
                     fontsize=10, fontweight='bold')
        ax.set_ylabel(f'Axe {axe2} ({self.inertie_pct[axe2-1]:.2f}%)', 
                     fontsize=10, fontweight='bold')
        ax.set_zlabel(f'Axe {axe3} ({self.inertie_pct[axe3-1]:.2f}%)', 
                     fontsize=10, fontweight='bold')
        ax.set_title('Projection 3D des Individus', fontsize=14, fontweight='bold', pad=20)
        ax.view_init(elev=rotation[0], azim=rotation[1])
        
        plt.tight_layout()
        plt.show()
    
    def plot_modalites_par_variable_3d(self, axes=(1, 2, 3), rotation=(30, 45)):
        """Projette les modalités en 3D séparées par variable"""
        axe1, axe2, axe3 = axes
        
        n_vars = len(self.vars_quali_toutes)
        n_cols = 3
        n_rows = (n_vars + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(18, 6*n_rows))
        
        for idx, var in enumerate(self.vars_quali_toutes):
            ax = fig.add_subplot(n_rows, n_cols, idx+1, projection='3d')
            cols = self.modalites_info[var]['colonnes']
            
            x = self.coord_modalites.loc[cols, f'Axe{axe1}'].values
            y = self.coord_modalites.loc[cols, f'Axe{axe2}'].values
            z = self.coord_modalites.loc[cols, f'Axe{axe3}'].values
            
            # Colorier par effectif
            effectifs = self.n_j[cols].values
            scatter = ax.scatter(x, y, z, s=200, c=effectifs, cmap='viridis', 
                               alpha=0.7, edgecolors='black', linewidth=2)
            
            for j, col in enumerate(cols):
                modalite = col.replace(f'{var}_', '')
                ax.text(x[j], y[j], z[j], modalite, fontsize=8, 
                       ha='center', va='center', fontweight='bold')
            
            # Plans de référence
            ax.plot([0, 0], [0, 0], [z.min(), z.max()], 'k-', alpha=0.2, linewidth=0.5)
            ax.plot([0, 0], [y.min(), y.max()], [0, 0], 'k-', alpha=0.2, linewidth=0.5)
            ax.plot([x.min(), x.max()], [0, 0], [0, 0], 'k-', alpha=0.2, linewidth=0.5)
            
            fig.colorbar(scatter, ax=ax, label='Effectif', shrink=0.5, pad=0.1)
            
            ax.set_xlabel(f'Axe {axe1}', fontsize=8)
            ax.set_ylabel(f'Axe {axe2}', fontsize=8)
            ax.set_zlabel(f'Axe {axe3}', fontsize=8)
            ax.set_title(f'{var}\n(m_j = {self.modalites_info[var]["m_j"]})', 
                        fontsize=10, fontweight='bold')
            ax.view_init(elev=rotation[0], azim=rotation[1])
        
        plt.tight_layout()
        plt.show()
    
    def plot_biplot_3d(self, axes=(1, 2, 3), color_by=None, top_modalites=10, rotation=(30, 45)):
        """Biplot 3D : individus et modalités sur le même graphique"""
        axe1, axe2, axe3 = axes
        
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # 1. Individus
        x_ind = self.coord_individus[f'Axe{axe1}'].values
        y_ind = self.coord_individus[f'Axe{axe2}'].values
        z_ind = self.coord_individus[f'Axe{axe3}'].values
        
        if color_by:
            if color_by in self.data.columns:
                var_color = self.data[color_by]
            elif color_by in self.data_discretisee.columns:
                var_color = self.data_discretisee[color_by]
            else:
                var_color = None
            
            if var_color is not None:
                categories = var_color.unique()
                colors = plt.cm.Set1(np.linspace(0, 1, len(categories)))
                
                for i, cat in enumerate(categories):
                    mask = var_color == cat
                    ax.scatter(x_ind[mask], y_ind[mask], z_ind[mask], 
                              s=30, alpha=0.4, color=colors[i], 
                              edgecolors='black', linewidth=0.3, label=f'Ind: {cat}')
        else:
            ax.scatter(x_ind, y_ind, z_ind, s=30, alpha=0.4, 
                      color='lightblue', edgecolors='black', 
                      linewidth=0.3, label='Individus')
        
        # 2. Modalités (top contributrices)
        contrib_tot = (self.ctr_modalites[f'Axe{axe1}'] + 
                      self.ctr_modalites[f'Axe{axe2}'] + 
                      self.ctr_modalites[f'Axe{axe3}'])
        top_indices = contrib_tot.nlargest(top_modalites).index
        
        x_mod = self.coord_modalites.loc[top_indices, f'Axe{axe1}'].values
        y_mod = self.coord_modalites.loc[top_indices, f'Axe{axe2}'].values
        z_mod = self.coord_modalites.loc[top_indices, f'Axe{axe3}'].values
        
        ax.scatter(x_mod, y_mod, z_mod, s=200, alpha=0.9, 
                  color='red', edgecolors='black', linewidth=2, 
                  marker='^', label='Modalités')
        
        for idx in top_indices:
            pos = self.coord_modalites.index.get_loc(idx)
            x = self.coord_modalites.loc[idx, f'Axe{axe1}']
            y = self.coord_modalites.loc[idx, f'Axe{axe2}']
            z = self.coord_modalites.loc[idx, f'Axe{axe3}']
            ax.text(x, y, z, idx, fontsize=8, fontweight='bold', color='darkred')
        
        # Plans de référence
        ax.plot([0, 0], [0, 0], [z_ind.min(), z_ind.max()], 'k-', alpha=0.3, linewidth=0.5)
        ax.plot([0, 0], [y_ind.min(), y_ind.max()], [0, 0], 'k-', alpha=0.3, linewidth=0.5)
        ax.plot([x_ind.min(), x_ind.max()], [0, 0], [0, 0], 'k-', alpha=0.3, linewidth=0.5)
        
        ax.set_xlabel(f'Axe {axe1} ({self.inertie_pct[axe1-1]:.2f}%)', 
                     fontsize=10, fontweight='bold')
        ax.set_ylabel(f'Axe {axe2} ({self.inertie_pct[axe2-1]:.2f}%)', 
                     fontsize=10, fontweight='bold')
        ax.set_zlabel(f'Axe {axe3} ({self.inertie_pct[axe3-1]:.2f}%)', 
                     fontsize=10, fontweight='bold')
        ax.set_title('Biplot 3D : Individus et Modalités', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=9)
        ax.view_init(elev=rotation[0], azim=rotation[1])
        
        plt.tight_layout()
        plt.show()
    
    def animation_3d_interactive(self, axes=(1, 2, 3), plot_type='modalites'):
        """
        Crée une visualisation 3D interactive avec plusieurs angles de vue
        plot_type: 'modalites', 'individus', ou 'biplot'
        """
        axe1, axe2, axe3 = axes
        
        # Créer 4 vues différentes
        angles = [(20, 30), (20, 120), (20, 210), (20, 300)]
        
        fig = plt.figure(figsize=(18, 14))
        
        for i, (elev, azim) in enumerate(angles):
            ax = fig.add_subplot(2, 2, i+1, projection='3d')
            
            if plot_type == 'modalites':
                x = self.coord_modalites[f'Axe{axe1}'].values
                y = self.coord_modalites[f'Axe{axe2}'].values
                z = self.coord_modalites[f'Axe{axe3}'].values
                
                contrib_tot = (self.ctr_modalites[f'Axe{axe1}'] + 
                              self.ctr_modalites[f'Axe{axe2}'] + 
                              self.ctr_modalites[f'Axe{axe3}'])
                
                scatter = ax.scatter(x, y, z, c=contrib_tot, s=100, 
                                   cmap='viridis', edgecolors='black', 
                                   linewidth=1, alpha=0.8)
                
                # Annoter top 5
                top5 = contrib_tot.nlargest(5).index
                for idx in top5:
                    pos = self.coord_modalites.index.get_loc(idx)
                    ax.text(x[pos], y[pos], z[pos], idx, fontsize=6, fontweight='bold')
                
                fig.colorbar(scatter, ax=ax, shrink=0.5, pad=0.05)
                ax.set_title(f'Vue {i+1}: Modalités (elev={elev}°, azim={azim}°)', 
                           fontsize=10, fontweight='bold')
            
            elif plot_type == 'individus':
                x = self.coord_individus[f'Axe{axe1}'].values
                y = self.coord_individus[f'Axe{axe2}'].values
                z = self.coord_individus[f'Axe{axe3}'].values
                
                cos2_sum = (self.cos2_individus[f'Axe{axe1}'] + 
                           self.cos2_individus[f'Axe{axe2}'] + 
                           self.cos2_individus[f'Axe{axe3}'])
                
                scatter = ax.scatter(x, y, z, c=cos2_sum, s=40, 
                                   cmap='YlOrRd', edgecolors='black', 
                                   linewidth=0.3, alpha=0.6)
                
                fig.colorbar(scatter, ax=ax, shrink=0.5, pad=0.05)
                ax.set_title(f'Vue {i+1}: Individus (elev={elev}°, azim={azim}°)', 
                           fontsize=10, fontweight='bold')
            
            # Plans de référence
            ax.plot([0, 0], [0, 0], [z.min(), z.max()], 'k-', alpha=0.2, linewidth=0.5)
            ax.plot([0, 0], [y.min(), y.max()], [0, 0], 'k-', alpha=0.2, linewidth=0.5)
            ax.plot([x.min(), x.max()], [0, 0], [0, 0], 'k-', alpha=0.2, linewidth=0.5)
            
            ax.set_xlabel(f'Axe {axe1}', fontsize=8)
            ax.set_ylabel(f'Axe {axe2}', fontsize=8)
            ax.set_zlabel(f'Axe {axe3}', fontsize=8)
            ax.view_init(elev=elev, azim=azim)
        
        plt.tight_layout()
        plt.show()
    
    def rapport_complet(self, axes_a_analyser=[(1, 2)], color_by=None, 
                       top_modalites=15, top_individus=10, graphes_3d=False):
        """Génère le rapport complet conforme au cours"""
        print("\n" + "="*80)
        print("GÉNÉRATION DU RAPPORT ACM COMPLET")
        print("="*80 + "\n")
        
        # 1. Tableau des inerties
        print("1. Tableau des inerties...")
        self.afficher_tableau_inerties()
        
        # 2. Scree plots
        print("2. Graphiques des valeurs propres...")
        self.plot_scree()
        
        # 3. η² (rapport de corrélation)
        print("3. Rapports de corrélation η²...")
        self.afficher_eta2()
        
        # 4. Pour chaque paire d'axes
        for axe1, axe2 in axes_a_analyser:
            print(f"\n{'='*80}")
            print(f"ANALYSE DES AXES {axe1} ET {axe2}")
            print(f"{'='*80}\n")
            
            # Contributions des modalités
            print(f"4.a) Top modalités - Axe {axe1}:")
            self.afficher_contributions_modalites(axe=axe1, top_n=top_modalites)
            
            print(f"4.b) Top modalités - Axe {axe2}:")
            self.afficher_contributions_modalites(axe=axe2, top_n=top_modalites)
            
            # Graphiques 2D
            print(f"5. Projection des modalités (Axes {axe1}-{axe2})...")
            self.plot_modalites(axes=(axe1, axe2), top_contrib=top_modalites)
            
            print(f"6. Modalités par variable (Axes {axe1}-{axe2})...")
            self.plot_modalites_par_variable(axes=(axe1, axe2))
            
            print(f"7. Projection des individus (Axes {axe1}-{axe2})...")
            self.plot_individus(axes=(axe1, axe2), color_by=color_by, 
                              top_contrib=top_individus)
            
            print(f"8. Cercle des variables (Axes {axe1}-{axe2})...")
            self.plot_variables_eta2(axes=(axe1, axe2))
        
        # 9. Graphiques 3D (si demandé)
        if graphes_3d:
            print("\n" + "="*80)
            print("VISUALISATIONS 3D")
            print("="*80 + "\n")
            
            print("9a. Projection 3D des modalités...")
            self.plot_modalites_3d(axes=(1, 2, 3), top_contrib=top_modalites)
            
            print("9b. Modalités par variable en 3D...")
            self.plot_modalites_par_variable_3d(axes=(1, 2, 3))
            
            print("9c. Projection 3D des individus...")
            self.plot_individus_3d(axes=(1, 2, 3), color_by=color_by, 
                                  top_contrib=top_individus)
            
            print("9d. Biplot 3D (individus + modalités)...")
            self.plot_biplot_3d(axes=(1, 2, 3), color_by=color_by, 
                               top_modalites=15)
            
            print("9e. Vues multiples 3D des modalités...")
            self.animation_3d_interactive(axes=(1, 2, 3), plot_type='modalites')
            
            print("9f. Vues multiples 3D des individus...")
            self.animation_3d_interactive(axes=(1, 2, 3), plot_type='individus')
        
        # 10. Vérification des relations barycentriques
        print("10. Vérification des relations barycentriques...")
        self.verifier_relations_barycentriques()
        
        print("\n" + "="*80)
        print("✓ RAPPORT ACM COMPLET GÉNÉRÉ")
        print("="*80)


# ========== EXEMPLE D'UTILISATION ==========
if __name__ == "__main__":
    print("\n" + "="*80)
    print("ACM CONFORME AU COURS - ROTTERDAM BREAST CANCER")
    print("="*80 + "\n")
    
    # ===== CHARGEMENT DES DONNÉES =====
    
    # --- Option 1: Rotterdam Breast Cancer ---
    fichier = "RotterdamBreastCancer_df.csv"
    df = pd.read_csv(fichier)
    
    # Supprimer les colonnes non utilisées
    if 'pid' in df.columns:
        df = df.drop(['pid', 'rtime', 'dtime'], axis=1, errors='ignore')
    
    vars_quantitatives = ['age', 'nodes', 'pgr', 'er', 'year']
    vars_qualitatives = ['meno', 'size', 'grade', 'hormon', 'chemo', 'recur', 'death']
    color_by = 'death'
    
    # --- Option 2: Heart Disease (Décommentez pour l'utiliser) ---
    # fichier = "heartdisease_tbl_df.csv"
    # df = pd.read_csv(fichier)
    # vars_quantitatives = ['Age', 'BP', 'Cholesterol', 'MaximumHR']
    # vars_qualitatives = ['Sex', 'ChestPain', 'BloodSugar', 
    #                     'ExerciseInducedAngina', 'HeartDisease']
    # color_by = 'HeartDisease'
    
    print("Aperçu des données:")
    print(df.head())
    print(f"\nDimensions: {df.shape}\n")
    
    # ===== CRÉATION DE L'ANALYSEUR ACM =====
    acm = ACM_Cours(
        data=df,
        vars_quanti=vars_quantitatives,
        vars_quali=vars_qualitatives
    )
    
    # ===== ÉTAPES DE L'ACM =====
    
    # Étape 1: Discrétisation
    acm.discretiser_variables(methode='quantiles', n_classes=4)
    
    # Étape 2: Préparation TDC
    acm.prepare_acm()
    
    # Étape 3: Calcul ACM
    acm.compute_acm(n_components=5)
    
    # Étape 4: Contributions et qualités
    acm.compute_contributions()
    
    # Étape 5: Rapport complet
    acm.rapport_complet(
        axes_a_analyser=[(1, 2), (1, 3), (2, 3)],
        color_by=color_by,
        top_modalites=20,
        top_individus=15,
        graphes_3d=True  # ✨ ACTIVER LES GRAPHIQUES 3D ✨
    )
    
    # ===== GRAPHIQUES 3D SUPPLÉMENTAIRES (utilisation directe) =====
    print("\n" + "="*80)
    print("GRAPHIQUES 3D SUPPLÉMENTAIRES")
    print("="*80 + "\n")
    
    # Exemple 1: Modalités 3D avec rotation personnalisée
    print("Modalités 3D - Vue personnalisée...")
    acm.plot_modalites_3d(axes=(1, 2, 3), top_contrib=15, rotation=(45, 60))
    
    # Exemple 2: Individus 3D
    print("Individus 3D avec coloration...")
    acm.plot_individus_3d(axes=(1, 2, 3), color_by=color_by, top_contrib=10, rotation=(30, 120))
    
    # Exemple 3: Biplot 3D
    print("Biplot 3D - Individus et modalités ensemble...")
    acm.plot_biplot_3d(axes=(1, 2, 3), color_by=color_by, top_modalites=12, rotation=(20, 45))
    
    # Exemple 4: Animation multi-vues
    print("Vues multiples des modalités...")
    acm.animation_3d_interactive(axes=(1, 2, 3), plot_type='modalites')
    
    print("Vues multiples des individus...")
    acm.animation_3d_interactive(axes=(1, 2, 3), plot_type='individus')
    
    # ===== RÉSULTATS NUMÉRIQUES SUPPLÉMENTAIRES =====
    print("\n" + "="*80)
    print("RÉSULTATS NUMÉRIQUES COMPLÉMENTAIRES")
    print("="*80 + "\n")
    
    print("1. INERTIE PAR VARIABLE:")
    print("-"*60)
    for var in acm.vars_quali_toutes:
        m_j = acm.modalites_info[var]['m_j']
        inertie_var = (1/acm.p) * (m_j - 1)
        print(f"{var:30s}: m_j = {m_j}, Inertie = {inertie_var:.4f}")
    
    print(f"\n{'Total':30s}: I_totale = {acm.I_totale_theorique:.4f}")
    
    print("\n2. TOP 10 CONTRIBUTIONS INDIVIDUS - AXE 1:")
    print("-"*60)
    top_ind = acm.ctr_individus['Axe1'].sort_values(ascending=False).head(10)
    for idx, val in top_ind.items():
        print(f"Individu {idx}: {val:.2f}%")
    
    print("\n3. EFFECTIFS DES MODALITÉS:")
    print("-"*60)
    print(acm.n_j.sort_values(ascending=False).head(15))
    
    print("\n" + "="*80)
    print("✓ ANALYSE ACM TERMINÉE")
    print("="*80)