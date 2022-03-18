# Name of the data
Data=T2spiral_aniso.nii.gz

# Global path to the folder containing the raw data
Path='/media/Disco1T/Granziera_data/INsIDER_1P_1C_2TP/Processed_Images'

# List of all the subjects
list_subjects="
INsIDER_C064
"

#Note that we are assuming this is the path to the image: Path_to_Data=$Path/$subi/MWF/$Data

# All the reconstructions will be saved within this folder by default
mkdir Recon_folder

for subi in $list_subjects; do
   echo "=========================== Subject:   " $subi " ==========================="
   Path_to_Data=$Path/$subi/MWF/$Data
   
   # Check if the subject's data exists"
   if [ -f $Path_to_Data ]  
   then
       #_____________ Run the preprocessing and reconstruction _____________#
       # Create list of processed subjects
       echo $subi >> Recon_folder/computed_subjects.txt

       mkdir Recon_folder/$subi

       echo "(1) Copy data to local folder"
       cp $Path_to_Data  Recon_folder/$subi/Data.nii.gz
   
       echo "(2) Brain extraction (BET) using FSL"
       fslmaths Recon_folder/$subi/Data.nii.gz -Tmean Recon_folder/$subi/Data_avg.nii.gz
       bet Recon_folder/$subi/Data_avg.nii.gz Recon_folder/$subi/Data_mask -m -v -f 0.5
       mv Recon_folder/$subi/Data_mask_mask.nii.gz Recon_folder/$subi/mask.nii.gz

       echo "(3) Parametric multicomponent T2 estimation (three compartments):"
       python3 call_estimation_script.py --path_to_folder='Recon_folder'/$subi/ --input='Data.nii.gz' --mask='mask.nii.gz'  --denoise='TV2D' --reg_para=1.5 --savefig='yes' --savefig_slice=17 --numcores=-1

       # Total water content/proton-density correction for bias-field inhomogeneity
       echo "(4) Bias-field correction of the proton density map (K_global) using FAST-FSL"
       # Estimate bias-field from the proton density map
       fast -t 3 -n 3 -H 0.1 -I 4 -l 20.0 -b -o Recon_folder/$subi/K_global Recon_folder/$subi/K_global
       # Apply the correction to get the corrected map (i.e., corr = Raw/field-map)
       fslmaths Recon_folder/$subi/K_global -div Recon_folder/$subi/K_global_bias Recon_folder/$subi/K_global_corrected
   else
       echo "Error: Data " $Path_to_Data " does not exist"
       echo $subi >> Recon_folder/subjects_with_problems.txt
   fi
done
