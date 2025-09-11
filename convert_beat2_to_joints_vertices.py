import os
import argparse
import numpy as np
import torch
import smplx
from tqdm import tqdm
import glob
from pathlib import Path


def pose_to_smplx_params(pose_165d):
    """Convert 165D pose to SMPL-X parameters"""
    if len(pose_165d.shape) == 1:
        pose_165d = pose_165d.reshape(1, -1)
    
    batch_size = pose_165d.shape[0]
    
    global_orient = pose_165d[:, :3]
    body_pose = pose_165d[:, 3:66]
    jaw_pose = pose_165d[:, 66:69]
    leye_pose = pose_165d[:, 69:72]
    reye_pose = pose_165d[:, 72:75]
    left_hand_pose = pose_165d[:, 75:120]
    right_hand_pose = pose_165d[:, 120:165]
    
    return {
        'global_orient': torch.tensor(global_orient).float(),
        'body_pose': torch.tensor(body_pose).float(),
        'jaw_pose': torch.tensor(jaw_pose).float(),
        'leye_pose': torch.tensor(leye_pose).float(),
        'reye_pose': torch.tensor(reye_pose).float(),
        'left_hand_pose': torch.tensor(left_hand_pose).float(),
        'right_hand_pose': torch.tensor(right_hand_pose).float(),
    }


def convert_poses_to_joints_vertices(poses, smplx_model, device='cpu', batch_size=64):
    """Convert pose parameters to joint positions and vertices using SMPL-X model
    
    Args:
        poses: numpy array of shape (T, 165) - T frames of 165D pose parameters
        smplx_model: loaded SMPL-X model
        device: computation device
        batch_size: batch size for processing (to avoid memory issues)
    
    Returns:
        joints: numpy array of shape (T, num_joints, 3)
        vertices: numpy array of shape (T, num_vertices, 3)
    """
    num_frames = poses.shape[0]
    all_joints = []
    all_vertices = []
    
    # Process frame by frame to avoid tensor size mismatch
    for i in range(num_frames):
        try:
            # Single frame processing
            frame_pose = poses[i:i+1]  # Keep batch dimension (1, 165)
            
            # Convert to SMPL-X parameters
            smplx_params = pose_to_smplx_params(frame_pose)
            
            # Move to device
            for key in smplx_params:
                smplx_params[key] = smplx_params[key].to(device)
            
            # Forward pass through SMPL-X
            with torch.no_grad():
                output = smplx_model(**smplx_params, return_verts=True)
                joints = output.joints.detach().cpu().numpy()  # (1, num_joints, 3)
                vertices = output.vertices.detach().cpu().numpy()  # (1, num_vertices, 3)
            
            all_joints.append(joints)
            all_vertices.append(vertices)
            
        except Exception as e:
            print(f"  Error processing frame {i}: {e}")
            # Create zero arrays as fallback
            if len(all_joints) > 0:
                zero_joints = np.zeros_like(all_joints[-1])
                zero_vertices = np.zeros_like(all_vertices[-1])
            else:
                # Initialize with reasonable sizes
                zero_joints = np.zeros((1, 127, 3))  # Approximate joint count
                zero_vertices = np.zeros((1, 10475, 3))  # Approximate vertex count
            all_joints.append(zero_joints)
            all_vertices.append(zero_vertices)
    
    # Concatenate all frames
    all_joints = np.concatenate(all_joints, axis=0)
    all_vertices = np.concatenate(all_vertices, axis=0)
    
    return all_joints, all_vertices


def process_beat2_file(input_file, output_file, smplx_model, device='cpu'):
    """Process a single BEAT2 file and save converted data"""
    try:
        # Load original data
        data = np.load(input_file)
        poses = data['poses']  # Shape: (T, 165)
        
        print(f"Processing {os.path.basename(input_file)}: {poses.shape[0]} frames")
        
        # Check for invalid poses
        if np.any(np.isnan(poses)):
            print(f"WARNING: NaN values found in {input_file}, replacing with zeros")
            poses = np.nan_to_num(poses)
        
        # Convert poses to joints and vertices
        joints, vertices = convert_poses_to_joints_vertices(poses, smplx_model, device)
        
        # Prepare output data (keep original data + add new fields)
        output_data = {}
        
        # Copy original fields
        for key in data.keys():
            output_data[key] = data[key]
        
        # Add new fields
        output_data['joints'] = joints.astype(np.float32)  # Shape: (T, num_joints, 3)
        output_data['vertices'] = vertices.astype(np.float32)  # Shape: (T, num_vertices, 3)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save converted data
        np.savez_compressed(output_file, **output_data)
        
        print(f"  Saved: joints {joints.shape}, vertices {vertices.shape}")
        return True
        
    except Exception as e:
        print(f"ERROR processing {input_file}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Convert BEAT2 pose parameters to joint positions and vertices')
    parser.add_argument('--beat2-dir', default='BEAT2', type=str,
                        help='Path to BEAT2 directory (e.g., BEAT2)')
    parser.add_argument('--smplx-model-dir', default='models_smplx_v1_1/models', type=str,
                        help='Path to SMPL-X model directory (e.g., models_smplx_v1_1/models)')
    parser.add_argument('--output-dir', default='BEAT2_joints_vertices', type=str,
                        help='Output directory for converted files')
    parser.add_argument('--language', default='all', type=str,
                        choices=['all', 'english', 'chinese', 'spanish', 'japanese'],
                        help='Which language subset to process')
    parser.add_argument('--gender', default='neutral', type=str,
                        choices=['neutral', 'male', 'female'],
                        help='SMPL-X model gender')
    parser.add_argument('--device', default='cuda', type=str,
                        help='Computation device (cuda/cpu)')
    parser.add_argument('--batch-size', default=32, type=int,
                        help='Batch size for processing')
    parser.add_argument('--max-files', default=None, type=int,
                        help='Maximum number of files to process (for testing)')
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    print(f"Using device: {args.device}")
    
    # Load SMPL-X model
    try:
        print(f"Loading SMPL-X model from {args.smplx_model_dir}...")
        smplx_model = smplx.create(
            args.smplx_model_dir, 
            model_type='smplx',
            gender=args.gender, 
            use_face_contour=False,
            use_pca=False,  # Use full hand pose (45D)
            ext='npz'
        )
        smplx_model = smplx_model.to(args.device)
        smplx_model.eval()
        print("✅ SMPL-X model loaded successfully!")
    except Exception as e:
        print(f"❌ ERROR: Failed to load SMPL-X model: {e}")
        return
    
    # Define language folders
    lang_folders = {
        'english': 'beat_english_v2.0.0',
        'chinese': 'beat_chinese_v2.0.0', 
        'spanish': 'beat_spanish_v2.0.0',
        'japanese': 'beat_japanese_v2.0.0'
    }
    
    # Determine which languages to process
    if args.language == 'all':
        languages_to_process = list(lang_folders.keys())
    else:
        languages_to_process = [args.language]
    
    print(f"Processing languages: {languages_to_process}")
    
    # Process each language
    total_processed = 0
    total_errors = 0
    
    for lang in languages_to_process:
        lang_folder = lang_folders[lang]
        input_lang_dir = os.path.join(args.beat2_dir, lang_folder, 'smplxflame_30')
        output_lang_dir = os.path.join(args.output_dir, lang_folder, 'smplxflame_30')
        
        if not os.path.exists(input_lang_dir):
            print(f"⚠️  Directory not found: {input_lang_dir}, skipping...")
            continue
        
        # Find all .npz files
        pose_files = glob.glob(os.path.join(input_lang_dir, '*.npz'))
        pose_files.sort()
        
        if args.max_files:
            pose_files = pose_files[:args.max_files]
        
        print(f"\n=== Processing {lang.upper()} ({len(pose_files)} files) ===")
        
        # Process each file
        for input_file in tqdm(pose_files, desc=f"Converting {lang}"):
            # Generate output filename
            rel_path = os.path.relpath(input_file, input_lang_dir)
            output_file = os.path.join(output_lang_dir, rel_path)
            
            # Skip if output file already exists
            if os.path.exists(output_file):
                print(f"  Skipping (exists): {os.path.basename(output_file)}")
                continue
            
            # Process file
            success = process_beat2_file(input_file, output_file, smplx_model, args.device)
            if success:
                total_processed += 1
            else:
                total_errors += 1
    
    # Print summary
    print(f"\n=== CONVERSION COMPLETE ===")
    print(f"✅ Successfully processed: {total_processed} files")
    print(f"❌ Errors: {total_errors} files")
    print(f"💾 Output directory: {args.output_dir}")
    
    # Print sample data info
    if total_processed > 0:
        print(f"\nOutput file structure:")
        print(f"  - Original fields: 'betas', 'poses', 'expressions', 'trans', etc.")
        print(f"  - New fields: 'joints' (T, {smplx_model.J_regressor.shape[0]}, 3), 'vertices' (T, {smplx_model.faces.max()+1}, 3)")


if __name__ == '__main__':
    main()