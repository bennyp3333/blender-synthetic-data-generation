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
    of multiple models in random environments with random camera positions.
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

        # Get model configuration - now a list of model specifications
        self.models_config = config.get('models', [])
        if not self.models_config:
            raise ValueError("No models defined in configuration")
            
        # Initialize list to store model objects and their properties
        self.models = []

        # Load all models defined in config
        self.load_models()
        
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
        
        # Print initialization success
        print(f"\nSynthetic Data Generator initialized with configuration:")
        print(f"- Models: {', '.join([model['obj'].name for model in self.models])}")
        print(f"- Output directory: {self.output_dir}")
        print(f"- Number of images: {self.num_images}")
        print(f"- HDRI directory: {self.hdri_dir} ({'Not found' if not self.hdri_files else f'{len(self.hdri_files)} HDRIs found'})")
        print()

    def load_models(self):
        """Load all models defined in the configuration."""
        self.models = []
        
        # Collect all categories for tracking
        self.categories = []
        category_ids = {}  # To track assigned IDs
        next_category_id = 0
        
        for model_config in self.models_config:
            model_name = model_config.get('name')
            
            if not model_name:
                print(f"Warning: Model configuration missing name, skipping")
                continue
                
            # Try to find the model in the scene
            model_obj = None
            for obj in bpy.data.objects:
                if obj.name == model_name:
                    model_obj = obj
                    break
            
            if not model_obj:
                print(f"Warning: Model '{model_name}' not found in scene, skipping")
                continue
                
            # Get or assign category ID and name
            category_name = model_config.get('category', model_name)
            
            # If this category hasn't been seen before, assign a new ID
            if category_name not in category_ids:
                category_ids[category_name] = next_category_id
                next_category_id += 1
                # Add to categories list
                self.categories.append({
                    'id': category_ids[category_name],
                    'name': category_name
                })
            
            # Store model with its properties
            model_data = {
                'obj': model_obj,
                'category_id': category_ids[category_name],
                'category_name': category_name
            }
            
            # Add any extra properties from config
            for key, value in model_config.items():
                if key not in ['name', 'category']:
                    model_data[key] = value
                    
            self.models.append(model_data)
            print(f"Loaded model: {model_name} (Category: {category_name}, ID: {category_ids[category_name]})")
        
        print(f"Total models loaded: {len(self.models)}")
        categories_list = [f'{c["name"]} (ID: {c["id"]})' for c in self.categories]
        print(f"Categories: {', '.join(categories_list)}")
        
        if not self.models:
            raise ValueError("No valid models were found in the scene. Please check your configuration.")

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
            # Calculate center of all models as target
            if not self.models:
                print("No models to position camera for")
                return False
                
            # Use the first model as primary target
            target_pos = Vector((0, 0, 0))
            weights_sum = 0
            
            # Calculate weighted center of all models
            for model in self.models:
                model_obj = model['obj']
                # Use model's location with a weight
                weight = 1.0  # Default weight
                target_pos += model_obj.location * weight
                weights_sum += weight
                
            # Normalize by weight sum
            if weights_sum > 0:
                target_pos /= weights_sum
            else:
                # Fallback to first model if weights don't work
                target_pos = self.models[0]['obj'].location.copy()
            
            # Find maximum dimension across all models for distance calculation
            max_dim = 0.1
            min_dim = 0.1
            
            for model in self.models:
                model_obj = model['obj']
                if hasattr(model_obj, 'dimensions'):
                    model_dims = model_obj.dimensions
                    local_max_dim = max(model_dims)
                    local_min_dim = min([d for d in model_dims if d > 0] or [0.1])
                    max_dim = max(max_dim, local_max_dim)
                    min_dim = min(min_dim, local_min_dim) if local_min_dim > 0 else min_dim
                
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
                if random.random() < 0.2:
                    self.camera.data.dof.use_dof = True
                else:
                    self.camera.data.dof.use_dof = False
                self.camera.data.dof.focus_distance = distance
                self.camera.data.dof.aperture_fstop = random.uniform(1.4, 2.4)
                print(f"Using depth of field: {self.camera.data.dof.use_dof} with aperture f/{self.camera.data.dof.aperture_fstop:.1f}")
                
            # Adjust sensor size for perspective variation
            self.camera.data.sensor_width = random.uniform(20, 45)
            
            print(f"Camera at ({x:.2f}, {y:.2f}, {z:.2f}) with {self.camera.data.lens:.1f}mm lens")
            print(f"Target offset: ({offset_x:.2f}, {offset_y:.2f}, {offset_z:.2f}), Roll: {degrees(roll_angle):.1f}°")
            
            # Safety check - for each model
            if self.config.get('ensure_visible', True):
                visible_models = self.check_models_visibility()
                if not visible_models:
                    print("Warning: No models visible in frame")
                
            return True
            
        except Exception as e:
            print(f"Error setting camera: {e}")
            import traceback
            traceback.print_exc()
            return False

    def check_models_visibility(self):
        """Check if any parts of the models would be visible in the frame."""
        visible_models = []
        
        for model in self.models:
            model_obj = model['obj']
            
            try:
                # For mesh objects, check vertices
                if model_obj.type == 'MESH':
                    # Get model vertices in world space
                    mesh = model_obj.data
                    matrix_world = model_obj.matrix_world
                    
                    # Sample some vertices (for efficiency)
                    vertices = []
                    if len(mesh.vertices) > 0:
                        sample_size = min(50, len(mesh.vertices))
                        if sample_size > 0:
                            sample_indices = random.sample(range(len(mesh.vertices)), sample_size)
                        else:
                            sample_indices = []
                    else:
                        sample_indices = []
                    
                    for idx in sample_indices:
                        # Transform vertex to world space
                        vertex_world = matrix_world @ mesh.vertices[idx].co
                        vertices.append(vertex_world)
                        
                    # Also add object origin
                    vertices.append(model_obj.location)
                    
                    # Project vertices to camera
                    visible_points = 0
                    for point in vertices:
                        projection = self.project_point_to_camera(point)
                        if projection:
                            visible_points += 1
                    
                    visibility_percentage = (visible_points / len(vertices)) * 100
                    print(f"Model {model_obj.name} visibility check: {visible_points}/{len(vertices)} sample points visible ({visibility_percentage:.1f}%)")
                    
                    # Count model as visible if enough points are visible
                    if visible_points >= 3:
                        visible_models.append(model_obj.name)
                    else:
                        print(f"WARNING: Model {model_obj.name} may be poorly visible or out of frame!")
                        
            except Exception as e:
                print(f"Error checking visibility for model {model_obj.name}: {e}")
        
        return visible_models
    
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

            if random.random() < 0.5:
                width, height = height, width
            
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
        """Project a 3D point to 2D camera space with pixel coordinates."""
        try:
            scene = bpy.context.scene
            render = scene.render
            
            # Use Blender's built-in function to convert world coordinates to camera view
            co_2d = bpy_extras.object_utils.world_to_camera_view(scene, self.camera, Vector(point_3d))
            
            # co_2d returns normalized coordinates (0-1)
            x, y, z = co_2d
            
            # Convert normalized coordinates to pixel coordinates
            screen_x = x * render.resolution_x
            screen_y = (1 - y) * render.resolution_y  # Flip Y to match image coordinates (top-left origin)
            
            # Check if point is within frame and in front of camera
            if 0 <= x <= 1 and 0 <= y <= 1 and z > 0:
                return (screen_x, screen_y)
            else:
                return None
                
        except Exception as e:
            print(f"Error projecting point: {e}")
            return None
    
    def find_model_bounding_box(self, model_obj):
        """Calculate the 2D bounding box of a specific model in camera view with pixel coordinates."""
        try:
            scene = bpy.context.scene
            render = scene.render
            
            # Get render dimensions
            render_width = render.resolution_x
            render_height = render.resolution_y
            
            # For collecting valid vertices in screen space
            coords_2d = []
            
            # Get the mesh data from the model
            depsgraph = bpy.context.evaluated_depsgraph_get()
            obj_eval = model_obj.evaluated_get(depsgraph)
            mesh = obj_eval.to_mesh()
            
            # Transform each vertex to camera view and collect visible ones
            for vertex in mesh.vertices:
                # Convert vertex from object space to world space
                vert_world = model_obj.matrix_world @ vertex.co
                
                # Project the 3D point to camera space
                co_2d = bpy_extras.object_utils.world_to_camera_view(scene, self.camera, vert_world)
                x, y, z = co_2d
                
                # Only include points that are in front of the camera
                if z > 0:
                    # Convert normalized coordinates to pixel coordinates
                    screen_x = x * render_width
                    screen_y = (1 - y) * render_height  # Flip Y to match image coordinates (top-left origin)
                    
                    coords_2d.append((screen_x, screen_y))
            
            # Clean up the temporary mesh
            obj_eval.to_mesh_clear()
            
            # Return None if model is not visible
            if not coords_2d:
                print(f"Model {model_obj.name} not visible in camera view")
                return None
            
            # Calculate bounding box in pixel coordinates
            min_x = max(0, min(x for x, y in coords_2d))
            min_y = max(0, min(y for x, y in coords_2d))
            max_x = min(render_width, max(x for x, y in coords_2d))
            max_y = min(render_height, max(y for x, y in coords_2d))
            
            # Return None if bounding box has no area
            if min_x >= max_x or min_y >= max_y:
                print(f"Bounding box for {model_obj.name} has no area")
                return None
            
            # Calculate width, height and center
            width = max_x - min_x
            height = max_y - min_y
            center_x = min_x + width / 2
            center_y = min_y + height / 2
            
            print(f"Bounding box for {model_obj.name} calculated (pixel coordinates):")            
            print(f"  min_x: {min_x:.1f}, min_y: {min_y:.1f}")
            print(f"  max_x: {max_x:.1f}, max_y: {max_y:.1f}")
            print(f"  center_x: {center_x:.1f}, center_y: {center_y:.1f}")
            print(f"  width: {width:.1f}, height: {height:.1f}")
            
            return {
                'min_x': min_x,
                'min_y': min_y,
                'max_x': max_x,
                'max_y': max_y,
                'center_x': center_x,
                'center_y': center_y,
                'width': width,
                'height': height
            }
            
        except Exception as e:
            print(f"Error calculating bounding box for {model_obj.name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def find_bounding_boxes(self):
        """Calculate bounding boxes for all models in the scene."""
        bboxes = []
        
        for model_data in self.models:
            model_obj = model_data['obj']
            category_id = model_data.get('category_id', 0)
            category_name = model_data.get('category_name', model_obj.name)
            
            bbox = self.find_model_bounding_box(model_obj)
            if bbox:
                # Add model metadata to bbox data
                bbox['model_name'] = model_obj.name
                bbox['category_id'] = category_id
                bbox['category_name'] = category_name
                bbox['distance'] = (self.camera.location - model_obj.location).length
                bboxes.append(bbox)
        
        return bboxes
    
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

    def create_annotations(self, image_id, width, height, bboxes_data):
        """Create annotation data for each model in the image."""
        if not bboxes_data:
            return None
        
        try:
            annotations = []
            
            for i, bbox in enumerate(bboxes_data):
                # Format for detection datasets
                annotation = {
                    'id': f"{image_id}_{i}",
                    'image_id': image_id,
                    'image_width': width,
                    'image_height': height,
                    'bbox': {
                        'min_x': bbox['min_x'], 
                        'min_y': bbox['min_y'], 
                        'max_x': bbox['max_x'], 
                        'max_y': bbox['max_y'],
                        'width': bbox['width'], 
                        'height': bbox['height'], 
                        'center_x': bbox['center_x'], 
                        'center_y': bbox['center_y']
                    },
                    'category_id': bbox['category_id'],
                    'category_name': bbox['category_name'],
                    'distance': bbox['distance'],
                    'model_name': bbox['model_name']
                }
                annotations.append(annotation)
            
            return annotations
            
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
            
            # Calculate bounding boxes for all models
            bboxes_data = self.find_bounding_boxes()
            
            # Skip if no models are visible
            if not bboxes_data:
                print(f"No models visible in image {image_id}, skipping")
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
                
            # Create annotation for all models in the image
            annotations = self.create_annotations(image_id, width, height, bboxes_data)
            if annotations:
                # Add these annotations to the full list
                all_annotations.extend(annotations)
                
                # Also save individual annotation file with all models in this image
                indiv_anno_path = os.path.join(self.annot_dir, f"{image_id}.json")
                with open(indiv_anno_path, 'w') as f:
                    json.dump({
                        'image_id': image_id,
                        'image_width': width,
                        'image_height': height,
                        'annotations': annotations
                    }, f, indent=2)
                    
                successful_renders += 1
                print(f"Created annotations for {len(annotations)} models in image {image_id}")
            else:
                print(f"Failed to create annotations for image {image_id}")
                failed_renders += 1
            
            # Progress update
            progress_percent = (i+1)/self.num_images*100
            print(f"Progress: {i+1}/{self.num_images} ({progress_percent:.1f}%)")
            print(f"Success: {successful_renders}, Failed: {failed_renders}")
        
        print(f"\n=== DATASET GENERATION COMPLETE ===")
        print(f"Generated {successful_renders} valid images")
        print(f"Failed {failed_renders} images")
        print(f"Output directory: {self.output_dir}")


def main():
    """Main function to run the synthetic data generator."""
    print("\n=== SYNTHETIC DATA GENERATOR ===")
    
    # Configuration (modify these settings as needed)
    config = {
        'output_dir': 'synthetic_data_4',  # Relative to blend file
        'hdri_dir': 'hdris',  # Relative to blend file
        'num_images': 10,  # Number of images to generate
        'camera_min_distance': 0.25,
        'camera_max_distance': 0.5,
        'min_focal_length': 24,  # Min focal length in mm
        'max_focal_length': 50,  # Max focal length in mm
        'max_roll_angle': 20,        # Maximum roll angle in degrees
        'extreme_angles_prob': 0.1,  # Probability of extreme angles
        'ensure_visible': True,      # Check if model is visible
        'render_samples': 512,  # Cycles samples
        
        # Define multiple models - add your models here
        'models': [
            {
                'name': 'Red_Bull_Can_250ml_v1',  # Object name in Blender
                'category': 'RedBull'  # Category name for annotations
            },
            {
                'name': 'Test_Box',  # Example second model
                'category': 'Test'
            }
            # Add more models as needed
        ]
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