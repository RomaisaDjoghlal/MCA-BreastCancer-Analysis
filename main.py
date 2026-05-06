import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import warnings
import os
from pathlib import Path
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

# ============================================================================
# PARTIE 1: CRÉATION FICHIER NETTOYÉ
# ============================================================================

def creer_fichier_nettoye(fichier_source, fichier_sortie="data_clean.csv"):
    """
    Crée un nouveau fichier avec seulement les variables nécessaires et nettoyées
    """
    print("="*80)
    print("CRÉATION FICHIER DE DONNÉES NETTOYÉ")
    print("="*80 + "\n")
    
    # 1. Chargement
    print("1. CHARGEMENT FICHIER SOURCE")
    df = pd.read_csv(fichier_source)
    print(f"✓ Fichier chargé: {fichier_source}")
    print(f"✓ Dimensions initiales: {df.shape}")
    print(f"✓ Colonnes disponibles: {list(df.columns)}\n")
    
    # 2. Sélection des variables à garder
    print("2. SÉLECTION DES VARIABLES")
    variables_a_garder = ['age', 'meno', 'size', 'grade', 'hormon', 'chemo', 'recur']
    
    # Vérifier que toutes les colonnes existent
    colonnes_manquantes = [col for col in variables_a_garder if col not in df.columns]
    if colonnes_manquantes:
        print(f"⚠ ATTENTION: Colonnes manquantes: {colonnes_manquantes}")
        variables_a_garder = [col for col in variables_a_garder if col in df.columns]
    
    df_clean = df[variables_a_garder].copy()
    print(f"✓ Variables conservées: {variables_a_garder}")
    print(f"✓ Nouvelles dimensions: {df_clean.shape}\n")
    
    # 3. Analyse des valeurs manquantes
    print("3. VALEURS MANQUANTES")
    missing = df_clean.isnull().sum()
    if missing.sum() > 0:
        missing_df = pd.DataFrame({
            'Nombre': missing[missing > 0], 
            'Pourcentage': (missing[missing > 0]/len(df_clean)*100).round(2)
        })
        print(missing_df)
        print()
    else:
        print("✓ Aucune valeur manquante\n")
    
    # 4. Imputation des valeurs manquantes
    print("4. IMPUTATION DES VALEURS MANQUANTES")
    n_imputed = 0
    for col in df_clean.columns:
        if df_clean[col].isnull().any():
            if df_clean[col].dtype in ['int64', 'float64']:
                val = df_clean[col].median()
                df_clean[col].fillna(val, inplace=True)
                print(f"  ✓ {col}: médiane = {val:.2f}")
                n_imputed += 1
            else:
                val = df_clean[col].mode()[0]
                df_clean[col].fillna(val, inplace=True)
                print(f"  ✓ {col}: mode = {val}")
                n_imputed += 1
    
    if n_imputed == 0:
        print("  ✓ Aucune imputation nécessaire")
    print()
    
    # 5. Analyse des outliers pour age
    if 'age' in df_clean.columns:
        print("5. ANALYSE OUTLIERS (AGE)")
        Q1, Q3 = df_clean['age'].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
        outliers = df_clean[(df_clean['age'] < lower) | (df_clean['age'] > upper)]
        print(f"  Bornes IQR: [{lower:.1f}, {upper:.1f}]")
        print(f"  Outliers détectés: {len(outliers)} ({len(outliers)/len(df_clean)*100:.1f}%)")
        print("  → Conservés (données médicales réelles)\n")
        
        # Statistiques age
        print("6. STATISTIQUES AGE")
        print(df_clean['age'].describe())
        print()
    
    # 6. Vérification finale
    print("7. VÉRIFICATION FINALE")
    print(f"✓ Dimensions finales: {df_clean.shape}")
    print(f"✓ Valeurs manquantes: {df_clean.isnull().sum().sum()}")
    print(f"✓ Lignes complètes: {df_clean.dropna().shape[0]}")
    print()
    
    # 7. Affichage des premières lignes
    print("8. APERÇU DES DONNÉES")
    print(df_clean.head(10))
    print()
    
    # 8. Distribution des variables qualitatives
    print("9. DISTRIBUTION DES VARIABLES QUALITATIVES")
    print("-"*80)
    for col in df_clean.columns:
        if col != 'age' and df_clean[col].dtype in ['int64', 'float64', 'object']:
            print(f"\n{col.upper()}:")
            print(df_clean[col].value_counts().sort_index())
    print()
    
    # 9. Sauvegarde
    print("10. SAUVEGARDE")
    df_clean.to_csv(fichier_sortie, index=False)
    print(f"✓ Fichier sauvegardé: {fichier_sortie}")
    print(f"✓ {len(df_clean)} lignes × {len(df_clean.columns)} colonnes")
    
    print("\n" + "="*80)
    print("✓ FICHIER NETTOYÉ CRÉÉ AVEC SUCCÈS")
    print("="*80 + "\n")
    
    return df_clean


# ============================================================================
# PARTIE 2: CLASSE ACM COMPLÈTE AVEC CORRECTION BENZÉCRI
# ============================================================================

class ACM_BreastCancer:
    
    def __init__(self, data, dossier="graphes"):
        self.data = data.copy()
        self.n = len(data)
        self.dossier = dossier
        self._creer_dossiers()
        
        # Variables qualitatives (tout sauf age)
        self.vars_quali = [col for col in data.columns if col != 'age']
        
        print("="*80)
        print("ACM - ROTTERDAM BREAST CANCER")
        print("="*80)
        print(f"n = {self.n} individus")
        print(f"Variable quantitative: age")
        print(f"Variables qualitatives: {len(self.vars_quali)}")
        print(f"Liste: {self.vars_quali}")
        print(f"Dossier graphiques: {self.dossier}/\n")
    
    def _creer_dossiers(self):
        """Créer structure dossiers pour tous les graphiques"""
        dossiers = [
            self.dossier,
            f"{self.dossier}/00_Benzecri",
            f"{self.dossier}/01_Scree_Plot",
            f"{self.dossier}/02_Graphe_par_variable",
            f"{self.dossier}/03_Biplot",
            f"{self.dossier}/04_Modalites_Contributions",
            f"{self.dossier}/05_Modalites_Cos2",
            f"{self.dossier}/06_Individus_Contributions",
            f"{self.dossier}/07_Individus_Cos2",
            f"{self.dossier}/08_Export_CSV"
        ]
        for d in dossiers:
            Path(d).mkdir(parents=True, exist_ok=True)
    
    def discretiser_age(self, n_classes=4):
        """Discrétise age en quartiles"""
        print("="*80)
        print("DISCRÉTISATION AGE")
        print("="*80 + "\n")
        
        self.data_disc = self.data.copy()
        
        if 'age' in self.data.columns:
            self.data_disc['age_cat'], bins = pd.qcut(
                self.data['age'], q=n_classes,
                labels=[f'Q{i+1}' for i in range(n_classes)],
                retbins=True, duplicates='drop'
            )
            
            print(f"Classes: {n_classes}")
            print("Bornes:")
            for i in range(len(bins)-1):
                print(f"  Q{i+1}: [{bins[i]:.1f}, {bins[i+1]:.1f}]")
            
            print("\nDistribution:")
            print(self.data_disc['age_cat'].value_counts().sort_index())
            
            self.vars_toutes = self.vars_quali + ['age_cat']
        else:
            print("⚠ Variable 'age' non trouvée, pas de discrétisation")
            self.vars_toutes = self.vars_quali
        
        self.p = len(self.vars_toutes)
        print(f"\n✓ p = {self.p} variables pour l'ACM\n")
    
    def prepare_acm(self):
        """Tableau Disjonctif Complet"""
        print("="*80)
        print("TABLEAU DISJONCTIF COMPLET")
        print("="*80 + "\n")
        
        dummies = []
        self.mod_info = {}
        
        for var in self.vars_toutes:
            d = pd.get_dummies(self.data_disc[var], prefix=var, prefix_sep='_')
            dummies.append(d)
            self.mod_info[var] = {
                'modalites': list(self.data_disc[var].unique()),
                'm_j': len(d.columns),
                'colonnes': list(d.columns)
            }
        
        self.X = pd.concat(dummies, axis=1)
        self.J = sum([self.mod_info[v]['m_j'] for v in self.vars_toutes])
        self.n_j = self.X.sum(axis=0)
        self.I_tot = self.J / self.p - 1
        
        print(f"TDC: {self.X.shape}")
        print(f"  n = {self.n} individus")
        print(f"  p = {self.p} variables")
        print(f"  J = {self.J} modalités")
        print(f"  Inertie totale = {self.I_tot:.4f}\n")
    
    def compute_acm(self, n_comp=5):
        """Calcul ACM"""
        print("="*80)
        print("CALCUL ACM")
        print("="*80 + "\n")
        
        # Pondération
        X_pond = self.X.copy()
        for var in self.vars_toutes:
            m_j = self.mod_info[var]['m_j']
            cols = self.mod_info[var]['colonnes']
            X_pond[cols] = X_pond[cols] / np.sqrt(m_j)
        
        # ACP du tableau pondéré
        self.r = min(self.n - 1, self.J - self.p)
        n_comp = min(n_comp, self.r)
        
        self.pca = PCA(n_components=n_comp)
        self.F = self.pca.fit_transform(X_pond)
        self.lambda_k = self.pca.explained_variance_
        self.G = self.pca.components_.T * np.sqrt(self.lambda_k)
        
        # DataFrames
        self.coord_ind = pd.DataFrame(
            self.F, columns=[f'Axe{k+1}' for k in range(n_comp)],
            index=self.data.index
        )
        self.coord_mod = pd.DataFrame(
            self.G, columns=[f'Axe{k+1}' for k in range(n_comp)],
            index=X_pond.columns
        )
        
        # Inerties
        self.inertie_pct = (self.lambda_k / self.I_tot) * 100
        self.inertie_cum = np.cumsum(self.inertie_pct)
        
        print("Valeurs propres:")
        for k in range(len(self.lambda_k)):
            print(f"  Axe {k+1}: λ={self.lambda_k[k]:.4f} ({self.inertie_pct[k]:.2f}%)")
        print()
    
    def correction_benzecri(self):
        """Applique la correction de Benzécri"""
        print("="*80)
        print("CORRECTION DE BENZÉCRI")
        print("="*80 + "\n")
        
        # Valeurs propres corrigées
        self.lambda_corr = np.zeros_like(self.lambda_k)
        
        for k in range(len(self.lambda_k)):
            if self.lambda_k[k] > 1/self.p:
                self.lambda_corr[k] = ((self.p / (self.p - 1)) * 
                                       (self.lambda_k[k] - 1/self.p))**2
            else:
                self.lambda_corr[k] = 0
        
        # Inertie corrigée
        self.I_corr = np.sum(self.lambda_corr)
        self.inertie_corr_pct = (self.lambda_corr / self.I_corr) * 100
        self.inertie_corr_cum = np.cumsum(self.inertie_corr_pct)
        
        print("Valeurs propres corrigées:")
        print(f"Seuil 1/p = {1/self.p:.4f}\n")
        
        for k in range(len(self.lambda_k)):
            print(f"Axe {k+1}:")
            print(f"  λ original  = {self.lambda_k[k]:.4f}")
            print(f"  λ corrigé   = {self.lambda_corr[k]:.4f}")
            print(f"  % inertie   = {self.inertie_corr_pct[k]:.2f}%")
            print(f"  % cumulé    = {self.inertie_corr_cum[k]:.2f}%")
            print()
        
        print(f"Inertie totale corrigée: {self.I_corr:.4f}\n")
    
    def compute_contributions(self):
        """Contributions et Cos²"""
        print("="*80)
        print("CONTRIBUTIONS ET COS²")
        print("="*80 + "\n")
        
        n_axes = len(self.lambda_k)
        
        # Contributions individus
        self.ctr_ind = pd.DataFrame(index=self.data.index)
        for k in range(n_axes):
            self.ctr_ind[f'Axe{k+1}'] = (
                (self.F[:, k]**2) / (self.n * self.lambda_k[k]) * 100
            )
        
        # Contributions modalités
        self.ctr_mod = pd.DataFrame(index=self.coord_mod.index)
        for k in range(n_axes):
            self.ctr_mod[f'Axe{k+1}'] = (
                (self.n_j / (self.n * self.p)) * (self.G[:, k]**2) / self.lambda_k[k]
            ) * 100
        
        # Cos² individus
        dist_ind = np.sum(self.F**2, axis=1)
        self.cos2_ind = pd.DataFrame(index=self.data.index)
        for k in range(n_axes):
            self.cos2_ind[f'Axe{k+1}'] = self.F[:, k]**2 / dist_ind
        
        # Cos² modalités
        dist_mod = np.sum(self.G**2, axis=1)
        self.cos2_mod = pd.DataFrame(index=self.coord_mod.index)
        for k in range(n_axes):
            self.cos2_mod[f'Axe{k+1}'] = self.G[:, k]**2 / dist_mod
        
        print("✓ Contributions et Cos² calculés\n")
    
    # ========================================================================
    # GRAPHIQUES AMÉLIORÉS
    # ========================================================================
    
    def plot_benzecri(self):
        """Graphique de comparaison avec correction de Benzécri"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        
        n = len(self.lambda_k)
        labels = [f'{k+1}' for k in range(n)]
        x = np.arange(n)
        width = 0.35
        
        # 1. Comparaison valeurs propres
        ax1.bar(x - width/2, self.lambda_k, width, label='Original', 
               color='steelblue', alpha=0.8, edgecolor='black')
        ax1.bar(x + width/2, self.lambda_corr, width, label='Corrigé (Benzécri)', 
               color='coral', alpha=0.8, edgecolor='black')
        ax1.axhline(1/self.p, color='red', linestyle='--', linewidth=2,
                   label=f'Seuil 1/p = {1/self.p:.3f}')
        ax1.set_xlabel('Axes', fontweight='bold', fontsize=11)
        ax1.set_ylabel('λ', fontweight='bold', fontsize=11)
        ax1.set_title('Valeurs Propres: Original vs Corrigé', fontweight='bold', fontsize=13)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3, axis='y')
        
        # 2. Inertie expliquée (%)
        ax2.plot(labels, self.inertie_pct, marker='o', linewidth=2.5, 
                label='Original', color='steelblue', markersize=8)
        ax2.plot(labels, self.inertie_corr_pct, marker='s', linewidth=2.5, 
                label='Corrigé (Benzécri)', color='coral', markersize=8)
        ax2.set_xlabel('Axes', fontweight='bold', fontsize=11)
        ax2.set_ylabel('% Inertie', fontweight='bold', fontsize=11)
        ax2.set_title('Inertie Expliquée par Axe', fontweight='bold', fontsize=13)
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)
        
        # 3. Inertie cumulée (%)
        ax3.plot(labels, self.inertie_cum, marker='o', linewidth=2.5, 
                label='Original', color='darkgreen', markersize=8)
        ax3.fill_between(range(n), self.inertie_cum, alpha=0.2, color='green')
        ax3.plot(labels, self.inertie_corr_cum, marker='s', linewidth=2.5, 
                label='Corrigé (Benzécri)', color='darkorange', markersize=8)
        ax3.fill_between(range(n), self.inertie_corr_cum, alpha=0.2, color='orange')
        ax3.set_xlabel('Axes', fontweight='bold', fontsize=11)
        ax3.set_ylabel('% Inertie Cumulée', fontweight='bold', fontsize=11)
        ax3.set_title('Inertie Cumulée', fontweight='bold', fontsize=13)
        ax3.legend(fontsize=10)
        ax3.grid(alpha=0.3)
        
        # 4. Tableau comparatif
        ax4.axis('off')
        data = []
        for k in range(n):
            data.append([
                f'Axe {k+1}',
                f'{self.lambda_k[k]:.4f}',
                f'{self.lambda_corr[k]:.4f}',
                f'{self.inertie_pct[k]:.2f}%',
                f'{self.inertie_corr_pct[k]:.2f}%'
            ])
        
        table = ax4.table(cellText=data,
                         colLabels=['Axe', 'λ Orig.', 'λ Corr.', '% Orig.', '% Corr.'],
                         cellLoc='center', loc='center', 
                         colWidths=[0.15, 0.2, 0.2, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.2)
        
        for i in range(5):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax4.set_title('Récapitulatif: Original vs Corrigé (Benzécri)', 
                     fontweight='bold', fontsize=13, pad=20)
        
        plt.tight_layout()
        f = f"{self.dossier}/00_Benzecri/correction_benzecri.png"
        plt.savefig(f, dpi=300, bbox_inches='tight')
        print(f"✓ Sauvegardé: {f}")
        plt.close()
    
    def plot_scree(self):
        """Scree plot"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        n = len(self.lambda_k)
        labels = [f'{k+1}' for k in range(n)]
        
        ax1.bar(labels, self.lambda_k, color='steelblue', alpha=0.8)
        ax1.axhline(1/self.p, color='red', linestyle='--', linewidth=2,
                   label=f'Seuil 1/p = {1/self.p:.3f}')
        ax1.set_xlabel('Axes', fontweight='bold')
        ax1.set_ylabel('λ', fontweight='bold')
        ax1.set_title('Valeurs Propres', fontweight='bold', fontsize=14)
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        ax2.plot(labels, self.inertie_cum, marker='o', linewidth=2, color='darkgreen')
        ax2.fill_between(range(n), self.inertie_cum, alpha=0.3, color='green')
        ax2.set_xlabel('Axes', fontweight='bold')
        ax2.set_ylabel('% Inertie Cumulée', fontweight='bold')
        ax2.set_title('Inertie Cumulée', fontweight='bold', fontsize=14)
        ax2.grid(alpha=0.3)
        
        ax3.bar(labels, self.inertie_pct, color='orange', alpha=0.8)
        ax3.set_xlabel('Axes', fontweight='bold')
        ax3.set_ylabel('% Inertie', fontweight='bold')
        ax3.set_title('Inertie par Axe', fontweight='bold', fontsize=14)
        ax3.grid(alpha=0.3)
        
        ax4.axis('off')
        data = [[f'Axe {k+1}', f'{self.lambda_k[k]:.4f}',
                f'{self.inertie_pct[k]:.2f}%', f'{self.inertie_cum[k]:.2f}%']
                for k in range(n)]
        table = ax4.table(cellText=data,
                         colLabels=['Axe', 'λ', '% Inertie', '% Cumulé'],
                         cellLoc='center', loc='center', colWidths=[0.2, 0.25, 0.25, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax4.set_title('Récapitulatif', fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        f = f"{self.dossier}/01_Scree_Plot/scree_plot.png"
        plt.savefig(f, dpi=300, bbox_inches='tight')
        print(f"✓ Sauvegardé: {f}")
        plt.close()
    
    def plot_par_variable(self):
        """Graphe par variable"""
        n_vars = len(self.vars_toutes)
        n_cols = 3
        n_rows = (n_vars + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
        axes = axes.flatten() if n_vars > 1 else [axes]
        
        for idx, var in enumerate(self.vars_toutes):
            ax = axes[idx]
            cols = self.mod_info[var]['colonnes']
            
            x = self.coord_mod.loc[cols, 'Axe1'].values
            y = self.coord_mod.loc[cols, 'Axe2'].values
            eff = self.n_j[cols].values
            
            ax.axhline(0, color='k', lw=0.5, alpha=0.5)
            ax.axvline(0, color='k', lw=0.5, alpha=0.5)
            
            sc = ax.scatter(x, y, s=200, c=eff, cmap='viridis',
                           alpha=0.7, edgecolors='black', lw=2)
            
            for j, col in enumerate(cols):
                mod = col.replace(f'{var}_', '')
                ax.text(x[j], y[j], mod, fontsize=9, ha='center',
                       va='center', fontweight='bold')
            
            plt.colorbar(sc, ax=ax, label='Effectif')
            ax.set_xlabel(f'Axe 1 ({self.inertie_pct[0]:.1f}%)', fontsize=10)
            ax.set_ylabel(f'Axe 2 ({self.inertie_pct[1]:.1f}%)', fontsize=10)
            ax.set_title(f'{var} (m={self.mod_info[var]["m_j"]})',
                        fontsize=12, fontweight='bold')
            ax.grid(alpha=0.3)
        
        for idx in range(n_vars, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        f = f"{self.dossier}/02_Graphe_par_variable/par_variable.png"
        plt.savefig(f, dpi=300, bbox_inches='tight')
        print(f"✓ Sauvegardé: {f}")
        plt.close()
    
    def plot_biplot(self):
        """Biplot avec toutes les modalités"""
        fig, ax = plt.subplots(figsize=(16, 14))
        
        # Individus
        x_ind = self.coord_ind['Axe1'].values
        y_ind = self.coord_ind['Axe2'].values
        
        if 'death' in self.data.columns:
            for val, color, label in [(0, 'lightblue', 'Vivant'),
                                      (1, 'red', 'Décédé')]:
                mask = self.data['death'] == val
                ax.scatter(x_ind[mask], y_ind[mask], s=20, alpha=0.3,
                          color=color, edgecolors='k', lw=0.2, label=label)
        else:
            ax.scatter(x_ind, y_ind, s=20, alpha=0.3, color='gray',
                      edgecolors='k', lw=0.2, label='Individus')
        
        # Modalités
        x_mod = self.coord_mod['Axe1'].values
        y_mod = self.coord_mod['Axe2'].values
        contrib_tot = self.ctr_mod['Axe1'] + self.ctr_mod['Axe2']
        
        sc = ax.scatter(x_mod, y_mod, s=150, c=contrib_tot, 
                       cmap='RdYlGn_r', edgecolors='black', lw=2, 
                       alpha=0.9, marker='^', label='Modalités', zorder=10)
        
        for idx in self.coord_mod.index:
            pos = self.coord_mod.index.get_loc(idx)
            ax.text(x_mod[pos], y_mod[pos], idx, fontsize=7, 
                   fontweight='bold', color='darkgreen', zorder=11)
        
        ax.axhline(0, color='k', lw=1, alpha=0.5)
        ax.axvline(0, color='k', lw=1, alpha=0.5)
        
        plt.colorbar(sc, ax=ax, label='Contribution Totale (%)')
        ax.set_xlabel(f'Axe 1 ({self.inertie_pct[0]:.1f}%)', fontweight='bold', fontsize=12)
        ax.set_ylabel(f'Axe 2 ({self.inertie_pct[1]:.1f}%)', fontweight='bold', fontsize=12)
        ax.set_title('Biplot - Individus et Modalités', fontweight='bold', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        f = f"{self.dossier}/03_Biplot/biplot_complet.png"
        plt.savefig(f, dpi=300, bbox_inches='tight')
        print(f"✓ Sauvegardé: {f}")
        plt.close()
    
    # ========================================================================
    # HISTOGRAMMES MODALITÉS - CONTRIBUTIONS PAR AXE
    # ========================================================================
    
    def plot_modalites_contributions_axes(self):
        """Histogrammes horizontaux des contributions des modalités (TOP 20 par axe)"""
        print("\n4. Modalités - Contributions par axe (TOP 20)...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))
        
        # AXE 1
        ctr_axe1 = self.ctr_mod['Axe1'].sort_values(ascending=False).head(20)
        colors1 = plt.cm.Reds(np.linspace(0.4, 0.9, len(ctr_axe1)))
        
        y_pos = np.arange(len(ctr_axe1))
        ax1.barh(y_pos, ctr_axe1.values, color=colors1, edgecolor='black', linewidth=1.2)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(ctr_axe1.index, fontsize=10, fontweight='bold')
        ax1.set_xlabel('Contribution (%)', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Modalités', fontweight='bold', fontsize=12)
        ax1.set_title(f'TOP 20 Modalités - Contribution Axe 1\n({self.inertie_pct[0]:.1f}% inertie)', 
                     fontweight='bold', fontsize=14)
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)
        
        # Ligne moyenne
        mean1 = self.ctr_mod['Axe1'].mean()
        ax1.axvline(mean1, color='blue', linestyle='--', linewidth=2.5, 
                   label=f'Moyenne = {mean1:.2f}%', alpha=0.8)
        ax1.legend(fontsize=11)
        
        # Ajouter valeurs
        for i, v in enumerate(ctr_axe1.values):
            ax1.text(v + 0.3, i, f'{v:.2f}%', va='center', fontsize=9, fontweight='bold')
        
        # AXE 2
        ctr_axe2 = self.ctr_mod['Axe2'].sort_values(ascending=False).head(20)
        colors2 = plt.cm.Blues(np.linspace(0.4, 0.9, len(ctr_axe2)))
        
        y_pos2 = np.arange(len(ctr_axe2))
        ax2.barh(y_pos2, ctr_axe2.values, color=colors2, edgecolor='black', linewidth=1.2)
        ax2.set_yticks(y_pos2)
        ax2.set_yticklabels(ctr_axe2.index, fontsize=10, fontweight='bold')
        ax2.set_xlabel('Contribution (%)', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Modalités', fontweight='bold', fontsize=12)
        ax2.set_title(f'TOP 20 Modalités - Contribution Axe 2\n({self.inertie_pct[1]:.1f}% inertie)', 
                     fontweight='bold', fontsize=14)
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3)
        
        mean2 = self.ctr_mod['Axe2'].mean()
        ax2.axvline(mean2, color='blue', linestyle='--', linewidth=2.5, 
                   label=f'Moyenne = {mean2:.2f}%', alpha=0.8)
        ax2.legend(fontsize=11)
        
        for i, v in enumerate(ctr_axe2.values):
            ax2.text(v + 0.3, i, f'{v:.2f}%', va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        f = f"{self.dossier}/04_Modalites_Contributions/modalites_contrib_axes.png"
        plt.savefig(f, dpi=300, bbox_inches='tight')
        print(f"  ✓ Sauvegardé: {f}")
        plt.close()
    
    def plot_modalites_cos2_axes(self):
        """Histogrammes horizontaux des Cos² des modalités (TOP 20 par axe)"""
        print("\n5. Modalités - Cos² par axe (TOP 20)...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))
        
        # AXE 1
        cos2_axe1 = self.cos2_mod['Axe1'].sort_values(ascending=False).head(20)
        colors1 = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(cos2_axe1)))
        
        y_pos = np.arange(len(cos2_axe1))
        ax1.barh(y_pos, cos2_axe1.values, color=colors1, edgecolor='black', linewidth=1.2)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(cos2_axe1.index, fontsize=10, fontweight='bold')
        ax1.set_xlabel('Cos²', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Modalités', fontweight='bold', fontsize=12)
        ax1.set_title(f'TOP 20 Modalités - Cos² Axe 1\n({self.inertie_pct[0]:.1f}% inertie)', 
                     fontweight='bold', fontsize=14)
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)
        
        mean1 = self.cos2_mod['Axe1'].mean()
        ax1.axvline(mean1, color='darkgreen', linestyle='--', linewidth=2.5, 
                   label=f'Moyenne = {mean1:.3f}', alpha=0.8)
        ax1.legend(fontsize=11)
        
        for i, v in enumerate(cos2_axe1.values):
            ax1.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9, fontweight='bold')
        
        # AXE 2
        cos2_axe2 = self.cos2_mod['Axe2'].sort_values(ascending=False).head(20)
        colors2 = plt.cm.plasma(np.linspace(0.2, 0.9, len(cos2_axe2)))
        
        y_pos2 = np.arange(len(cos2_axe2))
        ax2.barh(y_pos2, cos2_axe2.values, color=colors2, edgecolor='black', linewidth=1.2)
        ax2.set_yticks(y_pos2)
        ax2.set_yticklabels(cos2_axe2.index, fontsize=10, fontweight='bold')
        ax2.set_xlabel('Cos²', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Modalités', fontweight='bold', fontsize=12)
        ax2.set_title(f'TOP 20 Modalités - Cos² Axe 2\n({self.inertie_pct[1]:.1f}% inertie)', 
                     fontweight='bold', fontsize=14)
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3)
        
        mean2 = self.cos2_mod['Axe2'].mean()
        ax2.axvline(mean2, color='darkgreen', linestyle='--', linewidth=2.5, 
                   label=f'Moyenne = {mean2:.3f}', alpha=0.8)
        ax2.legend(fontsize=11)
        
        for i, v in enumerate(cos2_axe2.values):
            ax2.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        f = f"{self.dossier}/05_Modalites_Cos2/modalites_cos2_axes.png"
        plt.savefig(f, dpi=300, bbox_inches='tight')
        print(f"  ✓ Sauvegardé: {f}")
        plt.close()
    
    # ========================================================================
    # HISTOGRAMMES INDIVIDUS - CONTRIBUTIONS PAR AXE (INTELLIGENTS)
    # ========================================================================
    
    def plot_individus_contributions_intelligents(self):
        """
        Histogrammes des contributions des individus:
        - Séparation forte contribution vs faible contribution
        - Gestion automatique si trop d'individus
        """
        print("\n6. Individus - Contributions par axe (intelligent)...")
        
        # Seuil contribution significative (>moyenne)
        mean_contrib_1 = 100 / self.n
        mean_contrib_2 = 100 / self.n
        
        # AXE 1
        ctr1 = self.ctr_ind['Axe1'].sort_values(ascending=False)
        high_contrib_1 = ctr1[ctr1 > mean_contrib_1]
        low_contrib_1 = ctr1[ctr1 <= mean_contrib_1]
        
        # AXE 2
        ctr2 = self.ctr_ind['Axe2'].sort_values(ascending=False)
        high_contrib_2 = ctr2[ctr2 > mean_contrib_2]
        low_contrib_2 = ctr2[ctr2 <= mean_contrib_2]
        
        # ===== GRAPHIQUE 1: FORTES CONTRIBUTIONS =====
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, max(10, len(high_contrib_1)*0.25)))
        
        # Axe 1 - Fortes contributions
        y_pos1 = np.arange(len(high_contrib_1))
        colors1 = plt.cm.Reds(np.linspace(0.5, 0.95, len(high_contrib_1)))
        
        ax1.barh(y_pos1, high_contrib_1.values, color=colors1, 
                edgecolor='black', linewidth=1.0, alpha=0.85)
        ax1.set_yticks(y_pos1)
        ax1.set_yticklabels(high_contrib_1.index, fontsize=8, fontweight='bold')
        ax1.set_xlabel('Contribution (%)', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Individus', fontweight='bold', fontsize=12)
        ax1.set_title(f'Individus FORTE Contribution Axe 1\n({self.inertie_pct[0]:.1f}% inertie) - {len(high_contrib_1)} individus > moyenne',
                     fontweight='bold', fontsize=13)
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)
        ax1.axvline(mean_contrib_1, color='blue', linestyle='--', linewidth=2.5,
                   label=f'Moyenne = {mean_contrib_1:.3f}%', alpha=0.8)
        ax1.legend(fontsize=10)
        
        # Axe 2 - Fortes contributions
        y_pos2 = np.arange(len(high_contrib_2))
        colors2 = plt.cm.Blues(np.linspace(0.5, 0.95, len(high_contrib_2)))
        
        ax2.barh(y_pos2, high_contrib_2.values, color=colors2,
                edgecolor='black', linewidth=1.0, alpha=0.85)
        ax2.set_yticks(y_pos2)
        ax2.set_yticklabels(high_contrib_2.index, fontsize=8, fontweight='bold')
        ax2.set_xlabel('Contribution (%)', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Individus', fontweight='bold', fontsize=12)
        ax2.set_title(f'Individus FORTE Contribution Axe 2\n({self.inertie_pct[1]:.1f}% inertie) - {len(high_contrib_2)} individus > moyenne',
                     fontweight='bold', fontsize=13)
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3)
        ax2.axvline(mean_contrib_2, color='blue', linestyle='--', linewidth=2.5,
                   label=f'Moyenne = {mean_contrib_2:.3f}%', alpha=0.8)
        ax2.legend(fontsize=10)
        
        plt.tight_layout()
        f1 = f"{self.dossier}/06_Individus_Contributions/individus_FORTE_contribution.png"
        plt.savefig(f1, dpi=300, bbox_inches='tight')
        print(f"  ✓ Sauvegardé: {f1}")
        plt.close()
        
        # ===== GRAPHIQUE 2: FAIBLES CONTRIBUTIONS (TOP 50) =====
        fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(22, 14))
        
        # Limiter à top 50 pour lisibilité
        low_contrib_1_top = low_contrib_1.head(50)
        low_contrib_2_top = low_contrib_2.head(50)
        
        # Axe 1 - Faibles contributions
        y_pos3 = np.arange(len(low_contrib_1_top))
        colors3 = plt.cm.Greys(np.linspace(0.3, 0.7, len(low_contrib_1_top)))
        
        ax3.barh(y_pos3, low_contrib_1_top.values, color=colors3,
                edgecolor='black', linewidth=0.8, alpha=0.7)
        ax3.set_yticks(y_pos3)
        ax3.set_yticklabels(low_contrib_1_top.index, fontsize=7)
        ax3.set_xlabel('Contribution (%)', fontweight='bold', fontsize=12)
        ax3.set_ylabel('Individus', fontweight='bold', fontsize=12)
        ax3.set_title(f'Individus FAIBLE Contribution Axe 1 (TOP 50)\n({len(low_contrib_1)} total ≤ moyenne)',
                     fontweight='bold', fontsize=13)
        ax3.invert_yaxis()
        ax3.grid(axis='x', alpha=0.3)
        ax3.axvline(mean_contrib_1, color='red', linestyle='--', linewidth=2,
                   label=f'Moyenne = {mean_contrib_1:.3f}%', alpha=0.8)
        ax3.legend(fontsize=10)
        
        # Axe 2 - Faibles contributions
        y_pos4 = np.arange(len(low_contrib_2_top))
        colors4 = plt.cm.Greys(np.linspace(0.3, 0.7, len(low_contrib_2_top)))
        
        ax4.barh(y_pos4, low_contrib_2_top.values, color=colors4,
                edgecolor='black', linewidth=0.8, alpha=0.7)
        ax4.set_yticks(y_pos4)
        ax4.set_yticklabels(low_contrib_2_top.index, fontsize=7)
        ax4.set_xlabel('Contribution (%)', fontweight='bold', fontsize=12)
        ax4.set_ylabel('Individus', fontweight='bold', fontsize=12)
        ax4.set_title(f'Individus FAIBLE Contribution Axe 2 (TOP 50)\n({len(low_contrib_2)} total ≤ moyenne)',
                     fontweight='bold', fontsize=13)
        ax4.invert_yaxis()
        ax4.grid(axis='x', alpha=0.3)
        ax4.axvline(mean_contrib_2, color='red', linestyle='--', linewidth=2,
                   label=f'Moyenne = {mean_contrib_2:.3f}%', alpha=0.8)
        ax4.legend(fontsize=10)
        
        plt.tight_layout()
        f2 = f"{self.dossier}/06_Individus_Contributions/individus_FAIBLE_contribution_top50.png"
        plt.savefig(f2, dpi=300, bbox_inches='tight')
        print(f"  ✓ Sauvegardé: {f2}")
        plt.close()
        
        print(f"  ✓ Axe 1: {len(high_contrib_1)} fortes contrib, {len(low_contrib_1)} faibles contrib")
        print(f"  ✓ Axe 2: {len(high_contrib_2)} fortes contrib, {len(low_contrib_2)} faibles contrib")
    
    def plot_individus_cos2_intelligents(self):
        """
        Histogrammes des Cos² des individus:
        - Séparation bonne vs mauvaise qualité de représentation
        """
        print("\n7. Individus - Cos² par axe (intelligent)...")
        
        # Seuils
        mean_cos2_1 = self.cos2_ind['Axe1'].mean()
        mean_cos2_2 = self.cos2_ind['Axe2'].mean()
        
        # AXE 1
        cos2_1 = self.cos2_ind['Axe1'].sort_values(ascending=False)
        good_1 = cos2_1[cos2_1 > mean_cos2_1]
        bad_1 = cos2_1[cos2_1 <= mean_cos2_1]
        
        # AXE 2
        cos2_2 = self.cos2_ind['Axe2'].sort_values(ascending=False)
        good_2 = cos2_2[cos2_2 > mean_cos2_2]
        bad_2 = cos2_2[cos2_2 <= mean_cos2_2]
        
        # ===== BONNE QUALITÉ =====
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, max(10, len(good_1)*0.25)))
        
        # Axe 1
        y_pos1 = np.arange(len(good_1))
        colors1 = plt.cm.YlGn(np.linspace(0.4, 0.95, len(good_1)))
        
        ax1.barh(y_pos1, good_1.values, color=colors1,
                edgecolor='black', linewidth=1.0, alpha=0.85)
        ax1.set_yticks(y_pos1)
        ax1.set_yticklabels(good_1.index, fontsize=8, fontweight='bold')
        ax1.set_xlabel('Cos²', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Individus', fontweight='bold', fontsize=12)
        ax1.set_title(f'Individus BONNE Qualité Axe 1 (Cos²)\n({len(good_1)} individus > moyenne)',
                     fontweight='bold', fontsize=13)
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)
        ax1.axvline(mean_cos2_1, color='darkgreen', linestyle='--', linewidth=2.5,
                   label=f'Moyenne = {mean_cos2_1:.3f}', alpha=0.8)
        ax1.legend(fontsize=10)
        
        # Axe 2
        y_pos2 = np.arange(len(good_2))
        colors2 = plt.cm.YlGn(np.linspace(0.4, 0.95, len(good_2)))
        
        ax2.barh(y_pos2, good_2.values, color=colors2,
                edgecolor='black', linewidth=1.0, alpha=0.85)
        ax2.set_yticks(y_pos2)
        ax2.set_yticklabels(good_2.index, fontsize=8, fontweight='bold')
        ax2.set_xlabel('Cos²', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Individus', fontweight='bold', fontsize=12)
        ax2.set_title(f'Individus BONNE Qualité Axe 2 (Cos²)\n({len(good_2)} individus > moyenne)',
                     fontweight='bold', fontsize=13)
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3)
        ax2.axvline(mean_cos2_2, color='darkgreen', linestyle='--', linewidth=2.5,
                   label=f'Moyenne = {mean_cos2_2:.3f}', alpha=0.8)
        ax2.legend(fontsize=10)
        
        plt.tight_layout()
        f1 = f"{self.dossier}/07_Individus_Cos2/individus_BONNE_qualite_cos2.png"
        plt.savefig(f1, dpi=300, bbox_inches='tight')
        print(f"  ✓ Sauvegardé: {f1}")
        plt.close()
        
        # ===== MAUVAISE QUALITÉ (TOP 50) =====
        fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(22, 14))
        
        bad_1_top = bad_1.head(50)
        bad_2_top = bad_2.head(50)
        
        # Axe 1
        y_pos3 = np.arange(len(bad_1_top))
        colors3 = plt.cm.OrRd(np.linspace(0.3, 0.7, len(bad_1_top)))
        
        ax3.barh(y_pos3, bad_1_top.values, color=colors3,
                edgecolor='black', linewidth=0.8, alpha=0.7)
        ax3.set_yticks(y_pos3)
        ax3.set_yticklabels(bad_1_top.index, fontsize=7)
        ax3.set_xlabel('Cos²', fontweight='bold', fontsize=12)
        ax3.set_ylabel('Individus', fontweight='bold', fontsize=12)
        ax3.set_title(f'Individus MAUVAISE Qualité Axe 1 (TOP 50)\n({len(bad_1)} total ≤ moyenne)',
                     fontweight='bold', fontsize=13)
        ax3.invert_yaxis()
        ax3.grid(axis='x', alpha=0.3)
        ax3.axvline(mean_cos2_1, color='red', linestyle='--', linewidth=2,
                   label=f'Moyenne = {mean_cos2_1:.3f}', alpha=0.8)
        ax3.legend(fontsize=10)
        
        # Axe 2
        y_pos4 = np.arange(len(bad_2_top))
        colors4 = plt.cm.OrRd(np.linspace(0.3, 0.7, len(bad_2_top)))
        
        ax4.barh(y_pos4, bad_2_top.values, color=colors4,
                edgecolor='black', linewidth=0.8, alpha=0.7)
        ax4.set_yticks(y_pos4)
        ax4.set_yticklabels(bad_2_top.index, fontsize=7)
        ax4.set_xlabel('Cos²', fontweight='bold', fontsize=12)
        ax4.set_ylabel('Individus', fontweight='bold', fontsize=12)
        ax4.set_title(f'Individus MAUVAISE Qualité Axe 2 (TOP 50)\n({len(bad_2)} total ≤ moyenne)',
                     fontweight='bold', fontsize=13)
        ax4.invert_yaxis()
        ax4.grid(axis='x', alpha=0.3)
        ax4.axvline(mean_cos2_2, color='red', linestyle='--', linewidth=2,
                   label=f'Moyenne = {mean_cos2_2:.3f}', alpha=0.8)
        ax4.legend(fontsize=10)
        
        plt.tight_layout()
        f2 = f"{self.dossier}/07_Individus_Cos2/individus_MAUVAISE_qualite_cos2_top50.png"
        plt.savefig(f2, dpi=300, bbox_inches='tight')
        print(f"  ✓ Sauvegardé: {f2}")
        plt.close()
        
        print(f"  ✓ Axe 1: {len(good_1)} bonne qualité, {len(bad_1)} mauvaise qualité")
        print(f"  ✓ Axe 2: {len(good_2)} bonne qualité, {len(bad_2)} mauvaise qualité")
    
    # ========================================================================
    # EXPORT CSV COMPLET
    # ========================================================================
    
    def export_resultats_complets(self):
        """Export CSV de tous les résultats"""
        print("\n8. Export CSV complet...")
        
        dossier = f"{self.dossier}/08_Export_CSV"
        Path(dossier).mkdir(parents=True, exist_ok=True)
        
        # ===== 1. INDIVIDUS =====
        df_ind = pd.DataFrame({
            'Individu': self.coord_ind.index,
            'Coord_Axe1': self.coord_ind['Axe1'].values,
            'Coord_Axe2': self.coord_ind['Axe2'].values,
            'Contrib_Axe1': self.ctr_ind['Axe1'].values,
            'Contrib_Axe2': self.ctr_ind['Axe2'].values,
            'Contrib_Totale': (self.ctr_ind['Axe1'] + self.ctr_ind['Axe2']).values,
            'Cos2_Axe1': self.cos2_ind['Axe1'].values,
            'Cos2_Axe2': self.cos2_ind['Axe2'].values,
            'Cos2_Total': (self.cos2_ind['Axe1'] + self.cos2_ind['Axe2']).values
        })
        df_ind = df_ind.sort_values('Contrib_Totale', ascending=False)
        
        f1 = f"{dossier}/individus_complet.csv"
        df_ind.to_csv(f1, index=False)
        print(f"  ✓ Individus: {f1}")
        
        # ===== 2. MODALITÉS =====
        df_mod = pd.DataFrame({
            'Modalite': self.coord_mod.index,
            'Coord_Axe1': self.coord_mod['Axe1'].values,
            'Coord_Axe2': self.coord_mod['Axe2'].values,
            'Contrib_Axe1': self.ctr_mod['Axe1'].values,
            'Contrib_Axe2': self.ctr_mod['Axe2'].values,
            'Contrib_Totale': (self.ctr_mod['Axe1'] + self.ctr_mod['Axe2']).values,
            'Cos2_Axe1': self.cos2_mod['Axe1'].values,
            'Cos2_Axe2': self.cos2_mod['Axe2'].values,
            'Cos2_Total': (self.cos2_mod['Axe1'] + self.cos2_mod['Axe2']).values,
            'Effectif': self.n_j.values
        })
        df_mod = df_mod.sort_values('Contrib_Totale', ascending=False)
        
        f2 = f"{dossier}/modalites_complet.csv"
        df_mod.to_csv(f2, index=False)
        print(f"  ✓ Modalités: {f2}")
        
        # ===== 3. VALEURS PROPRES =====
        df_vp = pd.DataFrame({
            'Axe': [f'Axe {k+1}' for k in range(len(self.lambda_k))],
            'Lambda_Original': self.lambda_k,
            'Lambda_Benzecri': self.lambda_corr,
            'Inertie_Pct_Original': self.inertie_pct,
            'Inertie_Pct_Benzecri': self.inertie_corr_pct,
            'Inertie_Cum_Original': self.inertie_cum,
            'Inertie_Cum_Benzecri': self.inertie_corr_cum
        })
        
        f3 = f"{dossier}/valeurs_propres.csv"
        df_vp.to_csv(f3, index=False)
        print(f"  ✓ Valeurs propres: {f3}")
        
        print(f"  ✓ 3 fichiers CSV exportés dans {dossier}/")
    
    # ========================================================================
    # RAPPORT COMPLET
    # ========================================================================
    
    def rapport_complet(self):
        """Génère tous les graphiques"""
        print("\n" + "="*80)
        print("GÉNÉRATION RAPPORT COMPLET ACM")
        print("="*80 + "\n")
        
        print("0. Correction de Benzécri...")
        self.plot_benzecri()
        
        print("\n1. Scree Plot...")
        self.plot_scree()
        
        print("\n2. Graphe par variable...")
        self.plot_par_variable()
        
        print("\n3. Biplot complet...")
        self.plot_biplot()
        
        # Nouveaux graphiques améliorés
        self.plot_modalites_contributions_axes()
        self.plot_modalites_cos2_axes()
        self.plot_individus_contributions_intelligents()
        self.plot_individus_cos2_intelligents()
        self.export_resultats_complets()
        
        print("\n" + "="*80)
        print("✓ RAPPORT COMPLET GÉNÉRÉ")
        print(f"✓ Tous les graphiques dans: {self.dossier}/")
        print("="*80)


# ============================================================================
# PARTIE 3: SCRIPT PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("ACM - ROTTERDAM BREAST CANCER DATASET")
    print("ANALYSE COMPLÈTE AVEC CORRECTION DE BENZÉCRI")
    print("="*80 + "\n")
    
    # ÉTAPE 1: Création fichier nettoyé
    print("ÉTAPE 1: CRÉATION FICHIER NETTOYÉ\n")
    fichier_source = "RotterdamBreastCancer_df.csv"
    fichier_clean = "data_clean.csv"
    
    df = creer_fichier_nettoye(fichier_source, fichier_clean)
    
    # ÉTAPE 2: Analyse ACM complète
    print("\nÉTAPE 2: ANALYSE ACM\n")
    
    acm = ACM_BreastCancer(data=df, dossier="graphes_acm")
    acm.discretiser_age(n_classes=4)
    acm.prepare_acm()
    acm.compute_acm(n_comp=5)
    acm.correction_benzecri()
    acm.compute_contributions()
    acm.rapport_complet()
    
    # ÉTAPE 3: Résultats numériques
    print("\n" + "="*80)
    print("RÉSULTATS NUMÉRIQUES - RÉSUMÉ")
    print("="*80 + "\n")
    
    print("=" * 70)
    print("CORRECTION DE BENZÉCRI")
    print("=" * 70)
    print(f"Inertie totale originale : {acm.I_tot:.4f}")
    print(f"Inertie totale corrigée  : {acm.I_corr:.4f}")
    print(f"Seuil 1/p : {1/acm.p:.4f}\n")
    
    print("=" * 70)
    print("TOP 10 MODALITÉS - CONTRIBUTION TOTALE (AXE 1 + AXE 2)")
    print("=" * 70)
    contrib_mod_tot = acm.ctr_mod['Axe1'] + acm.ctr_mod['Axe2']
    print(contrib_mod_tot.sort_values(ascending=False).head(10))
    
    print("\n" + "=" * 70)
    print("TOP 10 INDIVIDUS - CONTRIBUTION TOTALE (AXE 1 + AXE 2)")
    print("=" * 70)
    contrib_ind_tot = acm.ctr_ind['Axe1'] + acm.ctr_ind['Axe2']
    print(contrib_ind_tot.sort_values(ascending=False).head(10))
    
    print("\n" + "=" * 70)
    print("STATISTIQUES CONTRIBUTIONS INDIVIDUS")
    print("=" * 70)
    mean_contrib_1 = 100 / acm.n
    mean_contrib_2 = 100 / acm.n
    high_1 = (acm.ctr_ind['Axe1'] > mean_contrib_1).sum()
    high_2 = (acm.ctr_ind['Axe2'] > mean_contrib_2).sum()
    print(f"Axe 1: {high_1} individus > moyenne ({mean_contrib_1:.3f}%)")
    print(f"Axe 2: {high_2} individus > moyenne ({mean_contrib_2:.3f}%)")
    print(f"Total individus: {acm.n}")
    
    print("\n" + "="*80)
    print("✓ ANALYSE ACM TERMINÉE AVEC SUCCÈS")
    print(f"✓ Fichier nettoyé: {fichier_clean}")
    print(f"✓ Graphiques: {acm.dossier}/")
    print(f"✓ Structure:")
    print(f"  - 00_Benzecri/")
    print(f"  - 01_Scree_Plot/")
    print(f"  - 02_Graphe_par_variable/")
    print(f"  - 03_Biplot/")
    print(f"  - 04_Modalites_Contributions/")
    print(f"  - 05_Modalites_Cos2/")
    print(f"  - 06_Individus_Contributions/")
    print(f"  - 07_Individus_Cos2/")
    print(f"  - 08_Export_CSV/")
    print("="*80)