import bpy
import os
import random
import uuid
import json
import numpy as np
import bpy_extras
from mathutils import Vector, Euler, Matrix
from math import radians, degrees, pi

class SyntheticDataGenerator:
    """
    Generates synthetic training data for 2D object tracking by rendering images 
    of a model in random environments with random camera positions.
    """
    
    def __init__(self, config):
        # Configuration
        self.config = config
        self.model_name = config.get('model_name', 'MyModel')
        # Handle relative paths properly
        blend_file_path = bpy.data.filepath
        blend_dir = os.path.dirname(blend_file_path) if blend_file_path else bpy.app.tempdir

        # Convert relative paths to absolute paths
        self.output_dir = os.path.normpath(os.path.join(blend_dir, config.get('output_dir', 'synthetic_data')))
        self.hdri_dir = os.path.normpath(os.path.join(blend_dir, config.get('hdri_dir', 'hdris')))
        
        self.num_images = config.get('num_images', 10)
        
        # Get keypoints configuration
        self.keypoints = config.get('keypoints', {})
        
        # Create output directories
        self.images_dir = os.path.join(self.output_dir, 'images')
        self.annot_dir = os.path.join(self.output_dir, 'annotations')
        self.ensure_directory_exists(self.images_dir)
        self.ensure_directory_exists(self.annot_dir)
        
        # Load available HDRIs
        self.hdri_files = self.get_hdri_files()
        
        # Get scene objects
        try:
            self.camera = bpy.data.objects['Camera']
        except KeyError:
            raise ValueError("Camera not found in scene. Please add a camera.")
            
        self.model = bpy.data.objects.get(self.model_name)
        if not self.model:
            raise ValueError(f"Model '{self.model_name}' not found in the scene. Available objects: {', '.join([obj.name for obj in bpy.data.objects])}")
  
        # Validate keypoints
        self.validate_keypoints()

        # Add visual markers for keypoints if debug mode is enabled
        if self.config.get('debug_mode', False):
            self.add_debug_markers()
        
        # Print initialization success
        print(f"\nSynthetic Data Generator initialized with configuration:")
        print(f"- Model: {self.model_name}")
        print(f"- Output directory: {self.output_dir}")
        print(f"- Number of images: {self.num_images}")
        print(f"- HDRI directory: {self.hdri_dir} ({'Not found' if not self.hdri_files else f'{len(self.hdri_files)} HDRIs found'})")
        print(f"- Keypoints defined: {len(self.keypoints)}")
        print()

    def validate_keypoints(self):
        """Validate that all defined keypoints can be found in the scene."""
        for kp_name, kp_info in self.keypoints.items():
            # Check for object-based keypoints
            if 'object' in kp_info:
                obj_name = kp_info['object']
                if obj_name not in bpy.data.objects:
                    print(f"Warning: Keypoint '{kp_name}' references non-existent object '{obj_name}'")
            
            # Check for vertex-based keypoints
            elif 'vertex_group' in kp_info and 'vertex_index' in kp_info:
                vg_name = kp_info['vertex_group']
                vertex_idx = kp_info['vertex_index']
                
                # Check if the vertex group exists
                if self.model.type == 'MESH':
                    if vg_name not in self.model.vertex_groups:
                        print(f"Warning: Keypoint '{kp_name}' references non-existent vertex group '{vg_name}'")
                    else:
                        # Check if vertex index is valid
                        if vertex_idx >= len(self.model.data.vertices):
                            print(f"Warning: Keypoint '{kp_name}' references invalid vertex index {vertex_idx}")
                else:
                    print(f"Warning: Model '{self.model_name}' is not a mesh, cannot use vertex-based keypoints")
            
            # Check for empty-based keypoints
            elif 'empty' in kp_info:
                empty_name = kp_info['empty']
                if empty_name not in bpy.data.objects or bpy.data.objects[empty_name].type != 'EMPTY':
                    print(f"Warning: Keypoint '{kp_name}' references non-existent empty '{empty_name}'")
            
            # Check for bone-based keypoints
            elif 'armature' in kp_info and 'bone' in kp_info:
                armature_name = kp_info['armature']
                bone_name = kp_info['bone']
                
                if armature_name not in bpy.data.objects or bpy.data.objects[armature_name].type != 'ARMATURE':
                    print(f"Warning: Keypoint '{kp_name}' references non-existent armature '{armature_name}'")
                elif bone_name not in bpy.data.objects[armature_name].data.bones:
                    print(f"Warning: Keypoint '{kp_name}' references non-existent bone '{bone_name}'")
            
            # Invalid keypoint specification
            else:
                print(f"Warning: Keypoint '{kp_name}' has invalid specification: {kp_info}")
                print("A keypoint must reference either an 'object', 'empty', or 'vertex_group'+'vertex_index', or 'armature'+'bone'")

    def add_debug_markers(self):
        """Add visible markers at keypoint positions for visual debugging."""
        # Remove any existing debug markers
        for obj in bpy.data.objects:
            if obj.name.startswith("DEBUG_MARKER_"):
                bpy.data.objects.remove(obj)
        
        # Create new markers at keypoint positions
        for kp_name, kp_info in self.keypoints.items():
            pos_3d = self.get_keypoint_position(kp_info)
            if pos_3d:
                # Create empty as marker
                marker = bpy.data.objects.new(f"DEBUG_MARKER_{kp_name}", None)
                marker.empty_display_type = 'SPHERE'
                marker.empty_display_size = 0.02
                marker.location = pos_3d
                # Set color for visibility
                marker.color = (1.0, 0.0, 0.0, 1.0)  # Bright red
                bpy.context.collection.objects.link(marker)
                print(f"Added debug marker for '{kp_name}' at {pos_3d}")
            else:
                print(f"Could not place debug marker for '{kp_name}', position unknown")

    def get_hdri_files(self):
        """Get list of HDRI files from the HDRI directory."""
        if not self.hdri_dir or not os.path.exists(self.hdri_dir):
            print(f"Warning: HDRI directory '{self.hdri_dir}' does not exist or is not specified")
            return []
        
        hdri_files = [f for f in os.listdir(self.hdri_dir) 
                if f.lower().endswith(('.hdr', '.exr', '.hdri'))]
        
        if not hdri_files:
            print(f"Warning: No HDRI files found in '{self.hdri_dir}'")
        
        return hdri_files
                
    @staticmethod
    def ensure_directory_exists(dir_path):
        """Create directory if it doesn't exist."""
        try:
            os.makedirs(dir_path, exist_ok=True)
            print(f"Directory ensured: {dir_path}")
        except Exception as e:
            print(f"Error creating directory {dir_path}: {e}")
            raise
    
    def set_random_hdri(self):
        """Set a random HDRI environment with random brightness."""
        if not self.hdri_files:
            print("No HDRI files found, using default world settings")
            return False
        
        try:
            # Get the current scene
            scene = bpy.context.scene
            
            # Select random HDRI file
            hdri_file = random.choice(self.hdri_files)
            hdri_path = os.path.join(self.hdri_dir, hdri_file)
            
            # Check if the scene has a world, if not create one
            if not scene.world:
                print("Scene has no world, creating one")
                new_world = bpy.data.worlds.new("SyntheticWorld")
                scene.world = new_world
            
            # Get the current world from the scene
            world = scene.world
            
            # Set up world nodes for HDRI
            world.use_nodes = True
            nodes = world.node_tree.nodes
            links = world.node_tree.links
            
            # Clear existing nodes
            nodes.clear()
            
            # Add texture environment node
            tex_env = nodes.new('ShaderNodeTexEnvironment')
            try:
                # Try to load the image
                try:
                    # Check if already loaded
                    if hdri_file in bpy.data.images:
                        tex_env.image = bpy.data.images[hdri_file]
                        print(f"Using already loaded HDRI: {hdri_file}")
                    else:
                        # Load new image
                        print(f"Loading HDRI from: {hdri_path}")
                        if not os.path.exists(hdri_path):
                            print(f"HDRI file not found: {hdri_path}")
                            return False
                        
                        tex_env.image = bpy.data.images.load(hdri_path)
                        tex_env.image.name = hdri_file  # Name it for reuse
                except RuntimeError as e:
                    print(f"Runtime error loading HDRI: {e}")
                    return False
                    
            except Exception as e:
                print(f"Error setting up HDRI texture: {e}")
                return False
            
            # Add background node
            bg_node = nodes.new('ShaderNodeBackground')
            
            # Set random brightness
            brightness = random.uniform(0.5, 1.5)
            bg_node.inputs[1].default_value = brightness
            
            # Add output node
            output = nodes.new('ShaderNodeOutputWorld')
            
            # Connect nodes
            links.new(tex_env.outputs[0], bg_node.inputs[0])
            links.new(bg_node.outputs[0], output.inputs[0])
            
            print(f"Set environment to '{hdri_file}' with brightness {brightness:.2f}")
            return True
            
        except Exception as e:
            print(f"Error setting HDRI: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def set_random_camera(self):
        """Position camera randomly with advanced variation options."""
        try:
            # Get model position and dimensions
            target_pos = self.model.location.copy()
            
            # Get model size
            if hasattr(self.model, 'dimensions'):
                model_dims = self.model.dimensions
                max_dim = max(model_dims)
                min_dim = min([d for d in model_dims if d > 0] or [0.1])
            else:
                max_dim = 1.0
                min_dim = 0.1
                
            # Select a camera mode - different shot types
            camera_modes = {
                'normal': 0.5,      # Standard shot
                'close_up': 0.15,   # Close up shot
                'wide': 0.15,       # Wide angle shot
                'low_angle': 0.1,   # Low angle (looking up)
                'high_angle': 0.1   # High angle (looking down)
            }
            
            mode = random.choices(list(camera_modes.keys()), 
                                weights=list(camera_modes.values()), k=1)[0]
            print(f"Camera mode: {mode}")
            
            # Base parameters - will be modified by mode
            distance_min = self.config.get('camera_min_distance', 2.0)
            distance_max = self.config.get('camera_max_distance', 5.0)
            phi_min = radians(15)  # Vertical angle minimum (radians)
            phi_max = pi - radians(15)  # Vertical angle maximum (radians)
            roll_max = radians(self.config.get('max_roll_angle', 15))  # Base maximum roll angle (radians)
            offset_factor = 0.15  # Base offset factor (as proportion of distance)
            
            # Adjust parameters based on mode
            if mode == 'close_up':
                distance_min *= 0.6
                distance_max *= 0.6
                roll_max *= 1.5  # More roll for dramatic close-ups
                offset_factor = 0.1  # Less offset for close-ups
                
            elif mode == 'wide':
                distance_min *= 1.2
                distance_max *= 1.2
                offset_factor = 0.2  # More offset for wide shots
                
            elif mode == 'low_angle':
                phi_min = radians(30)  # Higher minimum angle (from below)
                phi_max = radians(70)
                roll_max = radians(30)  # More roll for dramatic low angle
                
            elif mode == 'high_angle':
                phi_min = radians(100)  # Lower maximum angle (from above)
                phi_max = radians(160)
                roll_max = radians(20)
                
            # Random distance based on adjusted parameters
            distance = random.uniform(distance_min, distance_max)
            
            # Random angles
            theta = random.uniform(0, 2 * pi)  # Horizontal angle - full range
            phi = random.uniform(phi_min, phi_max)  # Vertical angle - based on mode
            
            # Convert to Cartesian coordinates
            x = target_pos.x + distance * np.sin(phi) * np.cos(theta)
            y = target_pos.y + distance * np.sin(phi) * np.sin(theta)
            z = target_pos.z + distance * np.cos(phi)
            
            # Set camera position
            self.camera.location = Vector((x, y, z))
            
            # Calculate frame offset based on distance and model size
            offset_scale = max(distance * offset_factor, max_dim * 0.2)
                
            # Different offset distributions for different axes
            offset_x = random.uniform(-offset_scale, offset_scale)
            offset_y = random.uniform(-offset_scale, offset_scale)
            offset_z = random.uniform(-offset_scale * 0.5, offset_scale * 0.5)  # Less vertical variation
            
            # Calculate target point with offset
            target_with_offset = Vector((
                target_pos.x + offset_x,
                target_pos.y + offset_y,
                target_pos.z + offset_z
            ))
            
            # Point camera at target
            direction = target_with_offset - self.camera.location
            rot_quat = direction.to_track_quat('-Z', 'Y')
            rot_euler = rot_quat.to_euler()
            
            # Random roll angle based on mode
            roll_angle = random.uniform(-roll_max, roll_max)
                
            # Apply roll
            rot_euler.rotate_axis('Z', roll_angle)
            self.camera.rotation_euler = rot_euler
            
            # Select focal length based on camera mode
            if mode == 'close_up':
                focal_min = max(self.config.get('min_focal_length', 20), 35)
                focal_max = max(self.config.get('max_focal_length', 50), 85)
            elif mode == 'wide':
                focal_min = max(self.config.get('min_focal_length', 20) * 0.7, 14)
                focal_max = min(self.config.get('max_focal_length', 50) * 0.7, 28)
            else:
                focal_min = self.config.get('min_focal_length', 20)
                focal_max = self.config.get('max_focal_length', 50)
                
            # Set focal length
            self.camera.data.lens = random.uniform(focal_min, focal_max)
            
            # Adjust depth of field
            if hasattr(self.camera.data, 'dof'):
                self.camera.data.dof.use_dof = True
                self.camera.data.dof.focus_distance = distance
                self.camera.data.dof.aperture_fstop = random.uniform(1.4, 5.6)
                print(f"Using depth of field with f/{self.camera.data.dof.aperture_fstop:.1f}")
                
            # Adjust sensor size for perspective variation
            self.camera.data.sensor_width = random.uniform(20, 45)
            
            print(f"Camera at ({x:.2f}, {y:.2f}, {z:.2f}) with {self.camera.data.lens:.1f}mm lens")
            print(f"Target offset: ({offset_x:.2f}, {offset_y:.2f}, {offset_z:.2f}), Roll: {degrees(roll_angle):.1f}°")
            
            # Safety check - simulate rendering a few points to confirm model will be somewhat in frame
            if self.config.get('ensure_visible', True):
                self.check_model_visibility()
                
            return True
            
        except Exception as e:
            print(f"Error setting camera: {e}")
            import traceback
            traceback.print_exc()
            return False

    def check_model_visibility(self):
        """Check if any part of the model would be visible in the frame."""
        try:
            # For mesh objects, check vertices
            if self.model.type == 'MESH':
                # Get model vertices in world space
                mesh = self.model.data
                matrix_world = self.model.matrix_world
                
                # Sample some vertices (for efficiency)
                vertices = []
                sample_size = min(50, len(mesh.vertices))  # Sample up to 50 vertices
                sample_indices = random.sample(range(len(mesh.vertices)), sample_size)
                
                for idx in sample_indices:
                    # Transform vertex to world space
                    vertex_world = matrix_world @ mesh.vertices[idx].co
                    vertices.append(vertex_world)
                    
                # Also add object origin and bounding box corners
                vertices.append(self.model.location)
                
                # Project vertices to camera
                visible_points = 0
                for point in vertices:
                    projection = self.project_point_to_camera(point)
                    if projection:
                        visible_points += 1
                
                visibility_percentage = (visible_points / len(vertices)) * 100
                print(f"Model visibility check: {visible_points}/{len(vertices)} sample points visible ({visibility_percentage:.1f}%)")
                
                # If almost no points are visible, warn but don't fail
                if visible_points < 3:
                    print("WARNING: Model may be poorly visible or out of frame!")
                    
        except Exception as e:
            print(f"Error checking model visibility: {e}")
            # Don't fail the camera setup if this check fails
    
    def set_random_resolution(self):
        """Set a random resolution for rendering."""
        try:
            # Common resolutions - using smaller ones by default for faster rendering
            resolutions = {
                "1280x720": 0.2,    # HD
                "800x600": 0.3,     # SVGA
                "640x480": 0.3,     # VGA
                "320x240": 0.2      # QVGA
            }
            
            # Select random resolution based on probabilities
            probabilities = list(resolutions.values())
            norm_probabilities = [p/sum(probabilities) for p in probabilities]
            chosen_res = random.choices(list(resolutions.keys()), 
                                        weights=norm_probabilities, k=1)[0]
            
            width, height = map(int, chosen_res.split('x'))
            
            scene = bpy.context.scene
            scene.render.resolution_x = width
            scene.render.resolution_y = height
            scene.render.resolution_percentage = 100
            
            print(f"Set resolution to {width}x{height}")
            
            return width, height
            
        except Exception as e:
            print(f"Error setting resolution: {e}")
            return 640, 480  # Default fallback
    
    def project_point_to_camera(self, point_3d):
        """Project a 3D point to 2D camera space with normalized coordinates."""
        try:
            scene = bpy.context.scene
            
            # Use Blender's built-in function to convert world coordinates to camera view
            co_2d = bpy_extras.object_utils.world_to_camera_view(scene, self.camera, Vector(point_3d))
            
            # co_2d returns normalized coordinates (0-1)
            x, y = co_2d.x, co_2d.y
            
            # Check if point is within frame
            if 0 <= x <= 1 and 0 <= y <= 1:
                return (x, y)
            else:
                return None
                
        except Exception as e:
            print(f"Error projecting point: {e}")
            return None
    
    def get_keypoint_position(self, kp_info):
        """Get the 3D world position of a keypoint based on its specification."""
        try:
            # Object-based keypoint
            if 'object' in kp_info:
                obj = bpy.data.objects.get(kp_info['object'])
                if obj:
                    # Get object origin or specific offset
                    if 'offset' in kp_info:
                        # Local offset from object origin
                        offset = Vector(kp_info['offset'])
                        # Transform offset to world space
                        world_pos = obj.matrix_world @ offset
                    else:
                        # Just use object origin
                        world_pos = obj.matrix_world.translation.copy()
                    return world_pos
            
            # Vertex-based keypoint
            elif 'vertex_group' in kp_info and 'vertex_index' in kp_info:
                if self.model.type == 'MESH':
                    vertex_idx = kp_info['vertex_index']
                    if vertex_idx < len(self.model.data.vertices):
                        # Get vertex position in local space
                        local_pos = self.model.data.vertices[vertex_idx].co
                        # Transform to world space
                        world_pos = self.model.matrix_world @ local_pos
                        return world_pos
            
            # Empty-based keypoint
            elif 'empty' in kp_info:
                empty = bpy.data.objects.get(kp_info['empty'])
                if empty and empty.type == 'EMPTY':
                    return empty.matrix_world.translation.copy()
            
            # Bone-based keypoint
            elif 'armature' in kp_info and 'bone' in kp_info:
                armature = bpy.data.objects.get(kp_info['armature'])
                if armature and armature.type == 'ARMATURE':
                    bone_name = kp_info['bone']
                    if bone_name in armature.data.bones:
                        bone = armature.data.bones[bone_name]
                        if 'head' in kp_info and kp_info['head']:
                            # Use bone head
                            pos_local = bone.head_local
                        else:
                            # Use bone tail (default)
                            pos_local = bone.tail_local
                        # Transform to world space
                        world_pos = armature.matrix_world @ pos_local
                        return world_pos
            
            # If we got here, keypoint couldn't be resolved
            return None
            
        except Exception as e:
            print(f"Error getting keypoint position: {e}")
            return None
        
    def get_keypoints_2d(self):
        """Get all keypoints projected to 2D camera view."""
        keypoints_2d = {}
        
        for kp_name, kp_info in self.keypoints.items():
            # Get 3D position
            pos_3d = self.get_keypoint_position(kp_info)
            if pos_3d is None:
                print(f"Could not determine 3D position for keypoint '{kp_name}'")
                keypoints_2d[kp_name] = {
                    'position': None,
                    'visible': False
                }
                continue
                
            # Project to 2D
            pos_2d = self.project_point_to_camera(pos_3d)
            
            # Store if visible
            if pos_2d is not None:
                # Convert to image coordinates for annotation
                keypoints_2d[kp_name] = {
                    'position': pos_2d,  # Normalized coordinates (x, y)
                    'visible': True
                }
                print(f"Keypoint '{kp_name}' is visible at normalized position {pos_2d}")
            else:
                # Keypoint is out of view or behind camera
                keypoints_2d[kp_name] = {
                    'position': None,
                    'visible': False
                }
                print(f"Keypoint '{kp_name}' is not visible")
                    
        return keypoints_2d

    def find_bounding_box(self):
        """Calculate the 2D bounding box of the model in camera view."""
        try:
            scene = bpy.context.scene
            
            # Get the camera matrix
            camera_matrix = self.camera.matrix_world.normalized().inverted()
            
            # Create a temporary mesh from the model
            depsgraph = bpy.context.evaluated_depsgraph_get()
            obj_eval = self.model.evaluated_get(depsgraph)
            mesh = obj_eval.to_mesh()
            mesh.transform(self.model.matrix_world)
            mesh.transform(camera_matrix)
            
            # Get camera frame in world space
            frame = [-v for v in self.camera.data.view_frame(scene=scene)[:3]]
            
            # Collect coordinates of vertices in camera view
            x_coords = []
            y_coords = []
            
            for vertex in mesh.vertices:
                co_local = vertex.co
                z = -co_local.z
                
                if z <= 0.0:
                    # Vertex is behind the camera
                    continue
                
                # Perspective division
                frame_scaled = [(v / (v.z / z)) for v in frame]
                
                min_x, max_x = frame_scaled[1].x, frame_scaled[2].x
                min_y, max_y = frame_scaled[0].y, frame_scaled[1].y
                
                # Convert to normalized coordinates
                x = (co_local.x - min_x) / (max_x - min_x)
                y = (co_local.y - min_y) / (max_y - min_y)
                
                x_coords.append(x)
                y_coords.append(y)
            
            # Remove the temporary mesh
            obj_eval.to_mesh_clear()
            
            # Return None if model is not visible
            if not x_coords or not y_coords:
                print("Model not visible in camera view")
                return None
            
            # Calculate and clip bounding box
            min_x = max(0.0, min(x_coords))
            min_y = max(0.0, min(y_coords))
            max_x = min(1.0, max(x_coords))
            max_y = min(1.0, max(y_coords))
            
            # Return None if bounding box has no area
            if min_x >= max_x or min_y >= max_y:
                print("Bounding box has no area")
                return None
            
            # Convert to [x, y, width, height] format
            width = max_x - min_x
            height = max_y - min_y
            center_x = min_x + width / 2
            center_y = min_y + height / 2
            
            print(f"Bounding box calculated: center=({center_x:.2f}, {center_y:.2f}), size=({width:.2f}, {height:.2f})")
            
            return {
                'bbox': [center_x, center_y, width, height],
                'bbox_corners': [(min_x, min_y), (max_x, max_y)]
            }
            
        except Exception as e:
            print(f"Error calculating bounding box: {e}")
            return None
    
    def render_image(self, output_path):
        """Render the current scene to an image file."""
        try:
            scene = bpy.context.scene
            scene.render.filepath = output_path
            scene.render.image_settings.file_format = 'PNG'
            
            # Set some reasonable render settings
            scene.render.engine = 'CYCLES'  # Use Cycles for better lighting
            scene.cycles.samples = self.config.get('render_samples', 16)  # Lower default for faster rendering
            scene.cycles.use_denoising = True
            
            print(f"Rendering image to {output_path} with {scene.cycles.samples} samples")
            
            # Render
            bpy.ops.render.render(write_still=True)
            
            print("Render complete")
            return True
            
        except Exception as e:
            print(f"Error rendering image: {e}")
            return False
    
    def create_annotations(self, image_id, width, height, bbox_data):
        """Create annotation data for the image."""
        if not bbox_data:
            return None
        
        try:
            # Get keypoints
            keypoints_2d = self.get_keypoints_2d()
            
            # Format for detection datasets (similar to COCO format)
            annotation = {
                'image_id': image_id,
                'image_width': width,
                'image_height': height,
                'bbox': bbox_data['bbox'],  # [x_center, y_center, width, height] (normalized)
                'bbox_corners': bbox_data['bbox_corners'],  # [(min_x, min_y), (max_x, max_y)] (normalized)
                'category_id': 0,  # Class ID (0 for the model)
                'category_name': self.model_name,
                'distance': (self.camera.location - self.model.location).length,
                'keypoints': keypoints_2d
            }
            
            return annotation
            
        except Exception as e:
            print(f"Error creating annotations: {e}")
            return None
    
    def generate_data(self):
        """Generate the synthetic dataset."""
        print(f"\n=== STARTING DATA GENERATION ===")
        print(f"Generating {self.num_images} synthetic images...")
        
        # Track all annotations
        all_annotations = []
        successful_renders = 0
        failed_renders = 0
        
        for i in range(self.num_images):
            # Generate unique ID for this image
            image_id = f"{i:06d}_{uuid.uuid4().hex[:8]}"
            
            print(f"\n--- Processing image {i+1}/{self.num_images}: {image_id} ---")
            
            # Set random environment
            self.set_random_hdri()
            
            # Set random camera position and parameters
            self.set_random_camera()
            
            # Set random resolution
            width, height = self.set_random_resolution()
            
            # Calculate bounding box
            bbox_data = self.find_bounding_box()
            
            # Skip if model not visible
            if not bbox_data:
                print(f"Model not visible in image {image_id}, skipping")
                failed_renders += 1
                continue
            
            # Prepare output paths
            image_path = os.path.join(self.images_dir, f"{image_id}.png")
            
            # Render image
            render_success = self.render_image(image_path)
            
            if not render_success:
                print(f"Failed to render image {image_id}")
                failed_renders += 1
                continue
                
            # Create annotation
            annotation = self.create_annotations(image_id, width, height, bbox_data)
            if annotation:
                all_annotations.append(annotation)
                
                # Also save individual annotation file
                indiv_anno_path = os.path.join(self.annot_dir, f"{image_id}.json")
                with open(indiv_anno_path, 'w') as f:
                    json.dump(annotation, f, indent=2)
                    
                successful_renders += 1
            else:
                print(f"Failed to create annotation for image {image_id}")
                failed_renders += 1
            
            # Progress update
            progress_percent = (i+1)/self.num_images*100
            print(f"Progress: {i+1}/{self.num_images} ({progress_percent:.1f}%)")
            print(f"Success: {successful_renders}, Failed: {failed_renders}")
        
        # Save all annotations to a single file
        all_anno_path = os.path.join(self.output_dir, "annotations.json")
        with open(all_anno_path, 'w') as f:
            json.dump({
                'images': all_annotations,
                'categories': [{
                    'id': 0,
                    'name': self.model_name
                }]
            }, f, indent=2)
        
        print(f"\n=== DATASET GENERATION COMPLETE ===")
        print(f"Generated {successful_renders} valid images")
        print(f"Failed {failed_renders} images")
        print(f"Output directory: {self.output_dir}")


def main():
    """Main function to run the synthetic data generator."""
    print("\n=== SYNTHETIC DATA GENERATOR ===")
    
    # Configuration (modify these settings as needed)
    config = {
        'model_name': 'Red_Bull_Can_250ml_v1',  # Change to your model name
        'output_dir': 'synthetic_data',  # Relative to blend file
        'hdri_dir': 'hdris',  # Relative to blend file
        'num_images': 5,  # Number of images to generate
        'camera_min_distance': 0.25,
        'camera_max_distance': 0.5,
        'min_focal_length': 24,  # Min focal length in mm
        'max_focal_length': 50,  # Max focal length in mm
        'max_roll_angle': 20,        # Maximum roll angle in degrees
        'extreme_angles_prob': 0.1,  # Probability of extreme angles
        'ensure_visible': True,      # Check if model is visible
        'render_samples': 1024,  # Cycles samples
        'debug_mode': False,
        
        # Define keypoints for tracking specific model parts
        'keypoints': {
            'can_top': {
                'object': 'Red_Bull_Can_250ml_v1',  # Using the main object
                'offset': [0, 0, 0.06],   # Offset from object origin to top (in local space)
                'description': 'Top of the can'
            },
            'can_bottom': {
                'empty': 'Can_Bottom',  # An empty at the bottom of the can
                'description': 'Bottom of the can'
            },
            
            # Example of vertex-based keypoint (if you have a mesh):
            # 'specific_feature': {
            #     'vertex_group': 'MainGroup',  # Vertex group name
            #     'vertex_index': 120,          # Index of vertex
            #     'description': 'A specific feature on the model'
            # },
            
            # Example of armature-based keypoint (if you have a rigged model):
            # 'bone_point': {
            #     'armature': 'BalloonRig',     # Armature object name
            #     'bone': 'Rope1',              # Bone name
            #     'head': True,                 # Use head of bone (default is tail)
            #     'description': 'Connection point of rope'
            # }
        }
    }
    
    # Print info about the scene
    print("\nScene Information:")
    print(f"Available objects: {', '.join([obj.name for obj in bpy.data.objects])}")
    
    # Create and run the generator
    try:
        generator = SyntheticDataGenerator(config)
        generator.generate_data()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease check the console for error details.")


if __name__ == "__main__":
    main()