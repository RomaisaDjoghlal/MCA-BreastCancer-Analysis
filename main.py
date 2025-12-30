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
            f"{self.dossier}/04_Ind_Ctr",
            f"{self.dossier}/05_Ind_cos",
            f"{self.dossier}/06_Modalite_cos",
            f"{self.dossier}/07_Mod_Ctr",
            f"{self.dossier}/08_Modalite_contribution",
            f"{self.dossier}/09_Cercle_correlation",
            f"{self.dossier}/10_Histogrammes",
            f"{self.dossier}/11_Biplot_complet",
            f"{self.dossier}/12_Contributions_individus"
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
    # GRAPHIQUES
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
        
        # Colorer l'en-tête
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
        
        # Valeurs propres
        ax1.bar(labels, self.lambda_k, color='steelblue', alpha=0.8)
        ax1.axhline(1/self.p, color='red', linestyle='--', linewidth=2,
                   label=f'Seuil 1/p = {1/self.p:.3f}')
        ax1.set_xlabel('Axes', fontweight='bold')
        ax1.set_ylabel('λ', fontweight='bold')
        ax1.set_title('Valeurs Propres', fontweight='bold', fontsize=14)
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Inertie cumulée
        ax2.plot(labels, self.inertie_cum, marker='o', linewidth=2, color='darkgreen')
        ax2.fill_between(range(n), self.inertie_cum, alpha=0.3, color='green')
        ax2.set_xlabel('Axes', fontweight='bold')
        ax2.set_ylabel('% Inertie Cumulée', fontweight='bold')
        ax2.set_title('Inertie Cumulée', fontweight='bold', fontsize=14)
        ax2.grid(alpha=0.3)
        
        # Inertie par axe
        ax3.bar(labels, self.inertie_pct, color='orange', alpha=0.8)
        ax3.set_xlabel('Axes', fontweight='bold')
        ax3.set_ylabel('% Inertie', fontweight='bold')
        ax3.set_title('Inertie par Axe', fontweight='bold', fontsize=14)
        ax3.grid(alpha=0.3)
        
        # Tableau
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
        """Biplot"""
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Individus par death
        x_ind = self.coord_ind['Axe1'].values
        y_ind = self.coord_ind['Axe2'].values
        
        if 'death' in self.data.columns:
            for val, color, label in [(0, 'lightblue', 'Vivant'),
                                      (1, 'red', 'Décédé')]:
                mask = self.data['death'] == val
                ax.scatter(x_ind[mask], y_ind[mask], s=30, alpha=0.4,
                          color=color, edgecolors='k', lw=0.3, label=label)
        else:
            ax.scatter(x_ind, y_ind, s=30, alpha=0.4, color='gray',
                      edgecolors='k', lw=0.3, label='Individus')
        
        # Top modalités
        contrib = self.ctr_mod['Axe1'] + self.ctr_mod['Axe2']
        top = contrib.nlargest(15).index
        
        x_mod = self.coord_mod.loc[top, 'Axe1'].values
        y_mod = self.coord_mod.loc[top, 'Axe2'].values
        
        ax.scatter(x_mod, y_mod, s=200, alpha=0.9, color='darkgreen',
                  edgecolors='k', lw=2, marker='^', label='Modalités')
        
        for idx in top:
            x = self.coord_mod.loc[idx, 'Axe1']
            y = self.coord_mod.loc[idx, 'Axe2']
            ax.text(x, y, idx, fontsize=8, fontweight='bold', color='darkgreen')
        
        ax.axhline(0, color='k', lw=1, alpha=0.5)
        ax.axvline(0, color='k', lw=1, alpha=0.5)
        ax.set_xlabel(f'Axe 1 ({self.inertie_pct[0]:.1f}%)', fontweight='bold')
        ax.set_ylabel(f'Axe 2 ({self.inertie_pct[1]:.1f}%)', fontweight='bold')
        ax.set_title('Biplot (Top 15 Modalités)', fontweight='bold', fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        f = f"{self.dossier}/03_Biplot/biplot.png"
        plt.savefig(f, dpi=300, bbox_inches='tight')
        print(f"✓ Sauvegardé: {f}")
        plt.close()
    
    def plot_ind_ctr(self):
        """Individus contribution"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        x = self.coord_ind['Axe1'].values
        y = self.coord_ind['Axe2'].values
        
        # Axe 1
        c1 = self.ctr_ind['Axe1']
        sc1 = ax1.scatter(x, y, c=c1, s=50, cmap='Reds',
                         edgecolors='k', lw=0.5, alpha=0.7)
        ax1.axhline(0, color='k', lw=0.5, alpha=0.5)
        ax1.axvline(0, color='k', lw=0.5, alpha=0.5)
        plt.colorbar(sc1, ax=ax1, label='Contribution (%)')
        ax1.set_xlabel(f'Axe 1 ({self.inertie_pct[0]:.1f}%)', fontweight='bold')
        ax1.set_ylabel(f'Axe 2 ({self.inertie_pct[1]:.1f}%)', fontweight='bold')
        ax1.set_title('Contribution Axe 1', fontweight='bold')
        ax1.grid(alpha=0.3)
        
        # Axe 2
        c2 = self.ctr_ind['Axe2']
        sc2 = ax2.scatter(x, y, c=c2, s=50, cmap='Blues',
                         edgecolors='k', lw=0.5, alpha=0.7)
        ax2.axhline(0, color='k', lw=0.5, alpha=0.5)
        ax2.axvline(0, color='k', lw=0.5, alpha=0.5)
        plt.colorbar(sc2, ax=ax2, label='Contribution (%)')
        ax2.set_xlabel(f'Axe 1 ({self.inertie_pct[0]:.1f}%)', fontweight='bold')
        ax2.set_ylabel(f'Axe 2 ({self.inertie_pct[1]:.1f}%)', fontweight='bold')
        ax2.set_title('Contribution Axe 2', fontweight='bold')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        f = f"{self.dossier}/04_Ind_Ctr/ind_ctr.png"
        plt.savefig(f, dpi=300, bbox_inches='tight')
        print(f"✓ Sauvegardé: {f}")
        plt.close()
    
    def plot_ind_cos(self):
        """Individus cos²"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        x = self.coord_ind['Axe1'].values
        y = self.coord_ind['Axe2'].values
        cos2 = self.cos2_ind['Axe1'] + self.cos2_ind['Axe2']
        
        sc = ax.scatter(x, y, c=cos2, s=50, cmap='YlOrRd',
                       edgecolors='k', lw=0.5, alpha=0.7)
        ax.axhline(0, color='k', lw=0.5, alpha=0.5)
        ax.axvline(0, color='k', lw=0.5, alpha=0.5)
        plt.colorbar(sc, ax=ax, label='Cos²')
        ax.set_xlabel(f'Axe 1 ({self.inertie_pct[0]:.1f}%)', fontweight='bold')
        ax.set_ylabel(f'Axe 2 ({self.inertie_pct[1]:.1f}%)', fontweight='bold')
        ax.set_title('Qualité Représentation (Cos²)', fontweight='bold', fontsize=14)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        f = f"{self.dossier}/05_Ind_cos/ind_cos2.png"
        plt.savefig(f, dpi=300, bbox_inches='tight')
        print(f"✓ Sauvegardé: {f}")
        plt.close()
    
    def plot_mod_cos(self):
        """Modalités cos²"""
        fig, ax = plt.subplots(figsize=(14, 12))
        
        x = self.coord_mod['Axe1'].values
        y = self.coord_mod['Axe2'].values
        cos2 = self.cos2_mod['Axe1'] + self.cos2_mod['Axe2']
        
        sc = ax.scatter(x, y, c=cos2, s=100, cmap='plasma',
                       edgecolors='k', lw=1, alpha=0.8)
        
        # Top 15
        top = cos2.nlargest(15).index
        for idx in top:
            pos = self.coord_mod.index.get_loc(idx)
            ax.text(x[pos], y[pos], idx, fontsize=7, fontweight='bold')
        
        ax.axhline(0, color='k', lw=0.5, alpha=0.5)
        ax.axvline(0, color='k', lw=0.5, alpha=0.5)
        plt.colorbar(sc, ax=ax, label='Cos²')
        ax.set_xlabel(f'Axe 1 ({self.inertie_pct[0]:.1f}%)', fontweight='bold')
        ax.set_ylabel(f'Axe 2 ({self.inertie_pct[1]:.1f}%)', fontweight='bold')
        ax.set_title('Modalités - Cos²', fontweight='bold', fontsize=14)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        f = f"{self.dossier}/06_Modalite_cos/mod_cos2.png"
        plt.savefig(f, dpi=300, bbox_inches='tight')
        print(f"✓ Sauvegardé: {f}")
        plt.close()
    
    def plot_mod_ctr(self):
        """Modalités contribution"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
        
        x = self.coord_mod['Axe1'].values
        y = self.coord_mod['Axe2'].values
        
        # Axe 1
        c1 = self.ctr_mod['Axe1']
        sc1 = ax1.scatter(x, y, c=c1, s=100, cmap='Reds',
                         edgecolors='k', lw=1, alpha=0.8)
        top1 = c1.nlargest(10).index
        for idx in top1:
            pos = self.coord_mod.index.get_loc(idx)
            ax1.text(x[pos], y[pos], idx, fontsize=7, fontweight='bold')
        
        ax1.axhline(0, color='k', lw=0.5, alpha=0.5)
        ax1.axvline(0, color='k', lw=0.5, alpha=0.5)
        plt.colorbar(sc1, ax=ax1, label='Contribution (%)')
        ax1.set_xlabel(f'Axe 1 ({self.inertie_pct[0]:.1f}%)', fontweight='bold')
        ax1.set_ylabel(f'Axe 2 ({self.inertie_pct[1]:.1f}%)', fontweight='bold')
        ax1.set_title('Modalités - Contribution Axe 1', fontweight='bold')
        ax1.grid(alpha=0.3)
        
        # Axe 2
        c2 = self.ctr_mod['Axe2']
        sc2 = ax2.scatter(x, y, c=c2, s=100, cmap='Blues',
                         edgecolors='k', lw=1, alpha=0.8)
        top2 = c2.nlargest(10).index
        for idx in top2:
            pos = self.coord_mod.index.get_loc(idx)
            ax2.text(x[pos], y[pos], idx, fontsize=7, fontweight='bold')
        
        ax2.axhline(0, color='k', lw=0.5, alpha=0.5)
        ax2.axvline(0, color='k', lw=0.5, alpha=0.5)
        plt.colorbar(sc2, ax=ax2, label='Contribution (%)')
        ax2.set_xlabel(f'Axe 1 ({self.inertie_pct[0]:.1f}%)', fontweight='bold')
        ax2.set_ylabel(f'Axe 2 ({self.inertie_pct[1]:.1f}%)', fontweight='bold')
        ax2.set_title('Modalités - Contribution Axe 2', fontweight='bold')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        f = f"{self.dossier}/07_Mod_Ctr/mod_ctr.png"
        plt.savefig(f, dpi=300, bbox_inches='tight')
        print(f"✓ Sauvegardé: {f}")
        plt.close()
    
    def plot_modalites_contribution(self):
        """Modalités colorées par contribution totale"""
        fig, ax = plt.subplots(figsize=(14, 12))
        
        x = self.coord_mod['Axe1'].values
        y = self.coord_mod['Axe2'].values
        contrib_tot = self.ctr_mod['Axe1'] + self.ctr_mod['Axe2']
        
        sc = ax.scatter(x, y, c=contrib_tot, s=150, cmap='RdYlGn_r',
                       edgecolors='black', lw=1.5, alpha=0.8)
        
        for idx in self.coord_mod.index:
            pos = self.coord_mod.index.get_loc(idx)
            ax.text(x[pos], y[pos], idx, fontsize=7, ha='center', 
                   va='center', fontweight='bold')
        
        ax.axhline(0, color='k', lw=1, alpha=0.5)
        ax.axvline(0, color='k', lw=1, alpha=0.5)
        
        plt.colorbar(sc, ax=ax, label='Contribution Totale (%)')
        ax.set_xlabel(f'Axe 1 ({self.inertie_pct[0]:.1f}%)', fontweight='bold', fontsize=12)
        ax.set_ylabel(f'Axe 2 ({self.inertie_pct[1]:.1f}%)', fontweight='bold', fontsize=12)
        ax.set_title('Modalités Colorées par Contribution Totale (Axe1 + Axe2)', 
                    fontweight='bold', fontsize=14)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        f = f"{self.dossier}/08_Modalite_contribution/mod_contrib_color.png"
        plt.savefig(f, dpi=300, bbox_inches='tight')
        print(f"✓ Sauvegardé: {f}")
        plt.close()
    
    def plot_cercle_correlation(self):
        """Cercle de corrélation avec toutes les modalités"""
        fig, ax = plt.subplots(figsize=(14, 14))
        
        circle = plt.Circle((0, 0), 1, color='navy', fill=False, linewidth=2, linestyle='--')
        ax.add_patch(circle)
        
        x = self.coord_mod['Axe1'].values
        y = self.coord_mod['Axe2'].values
        
        norms = np.sqrt(x**2 + y**2)
        x_norm = x / norms
        y_norm = y / norms
        
        contrib_tot = self.ctr_mod['Axe1'] + self.ctr_mod['Axe2']
        
        for i, idx in enumerate(self.coord_mod.index):
            ax.arrow(0, 0, x_norm[i]*0.95, y_norm[i]*0.95,
                    head_width=0.03, head_length=0.03, fc='gray', 
                    ec='gray', alpha=0.6, lw=1)
        
        sc = ax.scatter(x_norm, y_norm, c=contrib_tot, s=100, 
                       cmap='RdYlGn_r', edgecolors='black', lw=1.5, 
                       alpha=0.8, zorder=10)
        
        for i, idx in enumerate(self.coord_mod.index):
            ax.text(x_norm[i]*1.08, y_norm[i]*1.08, idx, 
                   fontsize=7, ha='center', va='center', fontweight='bold')
        
        ax.axhline(0, color='k', lw=1, alpha=0.5)
        ax.axvline(0, color='k', lw=1, alpha=0.5)
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        
        plt.colorbar(sc, ax=ax, label='Contribution Totale (%)')
        ax.set_xlabel(f'Axe 1 ({self.inertie_pct[0]:.1f}%)', fontweight='bold', fontsize=12)
        ax.set_ylabel(f'Axe 2 ({self.inertie_pct[1]:.1f}%)', fontweight='bold', fontsize=12)
        ax.set_title('Cercle de Corrélation - Toutes les Modalités', 
                    fontweight='bold', fontsize=14)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        f = f"{self.dossier}/09_Cercle_correlation/cercle_corr.png"
        plt.savefig(f, dpi=300, bbox_inches='tight')
        print(f"✓ Sauvegardé: {f}")
        plt.close()
    
    def plot_histogrammes_variables(self):
        """Histogrammes de chaque variable montrant les modalités"""
        n_vars = len(self.vars_toutes)
        n_cols = 3
        n_rows = (n_vars + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
        axes = axes.flatten() if n_vars > 1 else [axes]
        
        colors = plt.cm.Set3(np.linspace(0, 1, 12))
        
        for idx, var in enumerate(self.vars_toutes):
            ax = axes[idx]
            
            counts = self.data_disc[var].value_counts().sort_index()
            modalites = counts.index
            effectifs = counts.values
            
            bars = ax.bar(range(len(modalites)), effectifs, 
                         color=colors[:len(modalites)], 
                         edgecolor='black', linewidth=1.5, alpha=0.8)
            
            ax.set_xticks(range(len(modalites)))
            ax.set_xticklabels(modalites, rotation=45, ha='right')
            ax.set_xlabel('Modalités', fontweight='bold', fontsize=10)
            ax.set_ylabel('Effectif', fontweight='bold', fontsize=10)
            ax.set_title(f'{var} (n={len(modalites)} modalités)', 
                        fontweight='bold', fontsize=12)
            ax.grid(axis='y', alpha=0.3)
            
            for i, (bar, val) in enumerate(zip(bars, effectifs)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val}\n({val/self.n*100:.1f}%)',
                       ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        for idx in range(n_vars, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        f = f"{self.dossier}/10_Histogrammes/histogrammes_variables.png"
        plt.savefig(f, dpi=300, bbox_inches='tight')
        print(f"✓ Sauvegardé: {f}")
        plt.close()
    
    def plot_biplot_complet(self):
        """Biplot avec individus et toutes les modalités"""
        fig, ax = plt.subplots(figsize=(16, 14))
        
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
        ax.set_title('Biplot Complet - Individus et Toutes les Modalités', 
                    fontweight='bold', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        f = f"{self.dossier}/11_Biplot_complet/biplot_complet.png"
        plt.savefig(f, dpi=300, bbox_inches='tight')
        print(f"✓ Sauvegardé: {f}")
        plt.close()
    
    def plot_contributions_individus_axes(self):
        """Contributions des individus dans Axe 1 et Axe 2"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        
        ctr1 = self.ctr_ind['Axe1'].sort_values(ascending=False)
        top20_1 = ctr1.head(20)
        
        ax1.barh(range(len(top20_1)), top20_1.values, 
                color='crimson', edgecolor='black', linewidth=1.2, alpha=0.8)
        ax1.set_yticks(range(len(top20_1)))
        ax1.set_yticklabels(top20_1.index, fontsize=9)
        ax1.set_xlabel('Contribution (%)', fontweight='bold', fontsize=11)
        ax1.set_ylabel('Individus', fontweight='bold', fontsize=11)
        ax1.set_title(f'Top 20 Individus - Contribution Axe 1\n({self.inertie_pct[0]:.1f}% inertie)', 
                     fontweight='bold', fontsize=13)
        ax1.grid(axis='x', alpha=0.3)
        ax1.invert_yaxis()
        
        mean1 = 100 / self.n
        ax1.axvline(mean1, color='blue', linestyle='--', linewidth=2, 
                   label=f'Moyenne = {mean1:.3f}%')
        ax1.legend(fontsize=10)
        
        ctr2 = self.ctr_ind['Axe2'].sort_values(ascending=False)
        top20_2 = ctr2.head(20)
        
        ax2.barh(range(len(top20_2)), top20_2.values, 
                color='dodgerblue', edgecolor='black', linewidth=1.2, alpha=0.8)
        ax2.set_yticks(range(len(top20_2)))
        ax2.set_yticklabels(top20_2.index, fontsize=9)
        ax2.set_xlabel('Contribution (%)', fontweight='bold', fontsize=11)
        ax2.set_ylabel('Individus', fontweight='bold', fontsize=11)
        ax2.set_title(f'Top 20 Individus - Contribution Axe 2\n({self.inertie_pct[1]:.1f}% inertie)', 
                     fontweight='bold', fontsize=13)
        ax2.grid(axis='x', alpha=0.3)
        ax2.invert_yaxis()
        
        mean2 = 100 / self.n
        ax2.axvline(mean2, color='blue', linestyle='--', linewidth=2, 
                   label=f'Moyenne = {mean2:.3f}%')
        ax2.legend(fontsize=10)
        
        plt.tight_layout()
        f = f"{self.dossier}/12_Contributions_individus/contrib_ind_axes.png"
        plt.savefig(f, dpi=300, bbox_inches='tight')
        print(f"✓ Sauvegardé: {f}")
        plt.close()
    
    def rapport_complet(self):
        """Génère tous les graphiques"""
        print("\n" + "="*80)
        print("GÉNÉRATION RAPPORT COMPLET")
        print("="*80 + "\n")
        
        print("0. Correction de Benzécri...")
        self.plot_benzecri()
        
        print("\n1. Scree Plot...")
        self.plot_scree()
        
        print("\n2. Graphe par variable...")
        self.plot_par_variable()
        
        print("\n3. Biplot (Top 15)...")
        self.plot_biplot()
        
        print("\n4. Individus - Contribution...")
        self.plot_ind_ctr()
        
        print("\n5. Individus - Cos²...")
        self.plot_ind_cos()
        
        print("\n6. Modalités - Cos²...")
        self.plot_mod_cos()
        
        print("\n7. Modalités - Contribution...")
        self.plot_mod_ctr()
        
        print("\n8. Modalités colorées par contribution...")
        self.plot_modalites_contribution()
        
        print("\n9. Cercle de corrélation...")
        self.plot_cercle_correlation()
        
        print("\n10. Histogrammes des variables...")
        self.plot_histogrammes_variables()
        
        print("\n11. Biplot complet...")
        self.plot_biplot_complet()
        
        print("\n12. Contributions individus par axes...")
        self.plot_contributions_individus_axes()
        
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
    print("RÉSULTATS NUMÉRIQUES")
    print("="*80 + "\n")
    
    print("=" * 60)
    print("CORRECTION DE BENZÉCRI - RÉSUMÉ")
    print("=" * 60)
    print(f"Inertie totale originale : {acm.I_tot:.4f}")
    print(f"Inertie totale corrigée  : {acm.I_corr:.4f}")
    print(f"Seuil 1/p : {1/acm.p:.4f}\n")
    
    print("=" * 60)
    print("TOP 10 MODALITÉS - CONTRIBUTION AXE 1")
    print("=" * 60)
    print(acm.ctr_mod['Axe1'].sort_values(ascending=False).head(10))
    
    print("\n" + "=" * 60)
    print("TOP 10 MODALITÉS - CONTRIBUTION AXE 2")
    print("=" * 60)
    print(acm.ctr_mod['Axe2'].sort_values(ascending=False).head(10))
    
    print("\n" + "=" * 60)
    print("TOP 10 MODALITÉS - COS² (AXE1+AXE2)")
    print("=" * 60)
    cos2_tot = acm.cos2_mod['Axe1'] + acm.cos2_mod['Axe2']
    print(cos2_tot.sort_values(ascending=False).head(10))
    
    print("\n" + "=" * 60)
    print("TOP 10 INDIVIDUS - CONTRIBUTION AXE 1")
    print("=" * 60)
    print(acm.ctr_ind['Axe1'].sort_values(ascending=False).head(10))
    
    print("\n" + "=" * 60)
    print("TOP 10 INDIVIDUS - CONTRIBUTION AXE 2")
    print("=" * 60)
    print(acm.ctr_ind['Axe2'].sort_values(ascending=False).head(10))
    
    print("\n" + "="*80)
    print("✓ ANALYSE ACM TERMINÉE AVEC SUCCÈS")
    print(f"✓ Fichier nettoyé: {fichier_clean}")
    print(f"✓ Graphiques disponibles dans: {acm.dossier}/")
    print(f"✓ Total graphiques générés: 13 (dont correction Benzécri)")
    print("="*80)