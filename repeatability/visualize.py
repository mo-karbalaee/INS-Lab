import pandas
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


class vizualize_repeatability():
    def __init__(self, segments, data_dict, df_repeat, metrics_dict, metrics_df, save_path):
        self.segments = segments
        self.data_dict = data_dict
        self.df_repeat = df_repeat
        self.metrics_dict = metrics_dict
        self.metrics_df = metrics_df
        self.save_path = save_path

    def plot_heatmap_features_gestures_per_participant(self, save = False):
        # use cv
        # Vizualize: for each participant own plot:
        # heatmap on x-axis: gestures
        # y-axis features
        # z-axis CV (Color heatmap...)

        if self.metrics_df.empty:
            raise ValueError('metrics_dict is empty or invalid')

        for subject, subject_df in self.metrics_df.groupby('subject'):
            pivot = subject_df.pivot(index='feature', columns='gesture', values='cv')
            if pivot.empty:
                continue

            fig, ax = plt.subplots(figsize=(10, max(4, len(pivot) * 0.6)))
            im = ax.imshow(pivot, aspect='auto', cmap='viridis')

            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            ax.set_xlabel('Gesture')
            ax.set_ylabel('Feature')
            ax.set_title(f'{subject} - CV repeatability heatmap')

            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    value = pivot.iat[i, j]
                    if not pandas.isna(value):
                        ax.text(j, i, f'{value:.1f}', ha='center', va='center', color='white', fontsize=8)

            fig.colorbar(im, ax=ax, label='CV (%)')
            plt.tight_layout()
            
            if save:
                plt.savefig(f"{self.save_path}/heatmap_feat_gest_per_part_{subject}.png", dpi=300, bbox_inches='tight')
            
            plt.show()
            plt.close(fig)


    def plot_feature_repeat_ranking(self, save=False):
        # Feature repeatability table
        # Rows: subjects + mean row
        # Columns: features
        # Values: mean CV over gestures per feature
        
        if self.metrics_df.empty:
            raise ValueError('metrics_df is empty')
        
        # Group by subject and feature, aggregate over gestures
        feature_repeatability = self.metrics_df.groupby(['subject', 'feature'])['cv'].mean().unstack(fill_value=np.nan)
        
        # Add mean row
        mean_row = feature_repeatability.mean(axis=0)
        feature_repeatability = pandas.concat([feature_repeatability, pandas.DataFrame([mean_row], index=['Mean'])])
        
        # Create table plot
        fig, ax = plt.subplots(figsize=(max(10, len(feature_repeatability.columns) * 1.2), max(6, len(feature_repeatability) * 1.0)))
        ax.axis('tight')
        ax.axis('off')
        
        # Format data for display
        table_data = []
        for idx in feature_repeatability.index:
            row = [idx]
            for col in feature_repeatability.columns:
                val = feature_repeatability.loc[idx, col]
                row.append(f'{val:.2f}' if not pandas.isna(val) else 'N/A')
            table_data.append(row)
        
        columns = ['Subject'] + list(feature_repeatability.columns)
        table = ax.table(cellText=table_data, colLabels=columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        plt.title('Feature Repeatability (Mean CV over Gestures)', fontsize=14, pad=20)
        plt.tight_layout()
        
        if save:
            plt.savefig(f"{self.save_path}/table_features_repeatability.png", dpi=300, bbox_inches='tight')
        
        plt.show()
        plt.close(fig)




    def plot_gesture_repeat_ranking(self, save=False):
        # Gesture repeatability table
        # Rows: subjects + mean row
        # Columns: gestures
        # Values: mean CV over features per gesture
        
        if self.metrics_df.empty:
            raise ValueError('metrics_df is empty')
        
        # Group by subject and gesture, aggregate over features
        gesture_repeatability = self.metrics_df.groupby(['subject', 'gesture'])['cv'].mean().unstack(fill_value=np.nan)
        
        # Add mean row
        mean_row = gesture_repeatability.mean(axis=0)
        gesture_repeatability = pandas.concat([gesture_repeatability, pandas.DataFrame([mean_row], index=['Mean'])])
        
        # Create table plot
        fig, ax = plt.subplots(figsize=(max(10, len(gesture_repeatability.columns) * 1.2), max(6, len(gesture_repeatability) * 1.0)))
        ax.axis('tight')
        ax.axis('off')
        
        # Format data for display
        table_data = []
        for idx in gesture_repeatability.index:
            row = [idx]
            for col in gesture_repeatability.columns:
                val = gesture_repeatability.loc[idx, col]
                row.append(f'{val:.2f}' if not pandas.isna(val) else 'N/A')
            table_data.append(row)
        
        columns = ['Subject'] + list(gesture_repeatability.columns)
        table = ax.table(cellText=table_data, colLabels=columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        plt.title('Gesture Repeatability (Mean CV over Features)', fontsize=14, pad=20)
        plt.tight_layout()
        
        if save:
            plt.savefig(f"{self.save_path}/table_gestures_repeatability.png", dpi=300, bbox_inches='tight')
        
        plt.show()
        plt.close(fig)

    def plot_mixed_model_results(self, save=False):
        """
        Fit mixed linear model with fixed effects (Feature, Gesture) and random effect (Subject).
        Visualize results: coefficients table and forest plot.
        """
        if self.metrics_df.empty:
            raise ValueError('metrics_df is empty')
        
        # Prepare data for modeling
        model_data = self.metrics_df.copy()
        
        # Fit mixed linear model: cv ~ feature + gesture + (1|subject)
        model = smf.mixedlm("cv ~ C(feature) + C(gesture)", model_data, groups=model_data["subject"])
        result = model.fit()
        
        # Get fixed effects
        fixed_effects = result.fe_params
        fixed_effects_ci = result.conf_int().loc[fixed_effects.index]
        
        # Create summary table
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        table_data = [['Parameter', 'Estimate', '95% CI Lower', '95% CI Upper', 'Std. Error', 't-value', 'p-value']]
        
        for param in fixed_effects.index:
            estimate = fixed_effects[param]
            ci_lower = fixed_effects_ci.loc[param, 0]
            ci_upper = fixed_effects_ci.loc[param, 1]
            std_err = result.bse[param]
            t_val = result.tvalues[param]
            p_val = result.pvalues[param]
            
            table_data.append([
                param,
                f'{estimate:.4f}',
                f'{ci_lower:.4f}',
                f'{ci_upper:.4f}',
                f'{std_err:.4f}',
                f'{t_val:.4f}',
                f'{p_val:.4f}'
            ])
        
        table = ax.table(cellText=table_data, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header row
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title('Mixed Linear Model - Fixed Effects\n(cv ~ Feature + Gesture + (1|Subject))', 
                  fontsize=14, pad=20, weight='bold')
        plt.tight_layout()
        
        if save:
            plt.savefig(f"{self.save_path}/mixed_model_fixed_effects.png", dpi=300, bbox_inches='tight')
        
        plt.show()
        plt.close(fig)
        
        # Forest plot for fixed effects (excluding intercept)
        fixed_effects_no_intercept = fixed_effects[1:]
        ci_no_intercept = fixed_effects_ci.iloc[1:]
        
        fig, ax = plt.subplots(figsize=(10, len(fixed_effects_no_intercept) * 0.4 + 2))
        
        y_pos = np.arange(len(fixed_effects_no_intercept))[::-1]
        
        for i, (param, estimate) in enumerate(fixed_effects_no_intercept.items()):
            ci_lower = ci_no_intercept.loc[param, 0]
            ci_upper = ci_no_intercept.loc[param, 1]
            
            # Plot CI line
            ax.plot([ci_lower, ci_upper], [y_pos[i], y_pos[i]], 'b-', linewidth=2)
            # Plot point estimate
            ax.plot(estimate, y_pos[i], 'ro', markersize=8)
        
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([param for param in fixed_effects_no_intercept.index])
        ax.set_xlabel('Coefficient (95% CI)', fontsize=12)
        ax.set_title('Forest Plot - Fixed Effects Coefficients', fontsize=14, weight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f"{self.save_path}/mixed_model_forest_plot.png", dpi=300, bbox_inches='tight')
        
        plt.show()
        plt.close(fig)
        
        # Print model summary to console
        print("\n" + "="*80)
        print("MIXED LINEAR MODEL SUMMARY")
        print("="*80)
        print(result.summary())
        print("\nRandom Effects (Subject intercepts):")
        print(result.random_effects)
        print("="*80 + "\n")
        
        return result