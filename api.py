def FacemakerWebAPI(client, host: str = "0.0.0.0", port: int = 3223, debug: bool = False):
    """
    Start Client API server with all endpoints.
    
    Parameters:
    - client (Client): Facemaker Client instance
    - host (str): Host to run the server on
    - port (int): Port to run the server on
    - debug (bool): Enable Flask debug mode
    """
    try:
        from flask import Flask, request, jsonify, render_template, send_file
        import tempfile, os, mimetypes, shutil, atexit
        
        app = Flask(__name__)
        app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
        
        # Create a persistent temporary directory for this session
        TEMP_DIR = tempfile.mkdtemp(prefix='fm1_')
        
        # Cleanup function to remove temporary files on server shutdown
        def cleanup_temp_files():
            if os.path.exists(TEMP_DIR):
                shutil.rmtree(TEMP_DIR)
        
        # Register cleanup function
        atexit.register(cleanup_temp_files)
        
        def __send_file(file_path):
            """Send a file as binary response with proper mime type.
            
            Parameters:
                file_path: Path to the file to send
            Returns:
                Flask response object with binary file data
            """
            mime_type, _ = mimetypes.guess_type(file_path)
            return send_file(
                file_path,
                mimetype=mime_type,
                as_attachment=True,
                download_name=os.path.basename(file_path)
            )

        @app.route('/')
        def api_docs():
            return render_template('index.html')

        @app.route('/api/download/<path:filename>')
        def download_file(filename):
            """Download endpoint for retrieving processed files"""
            try:
                file_path = os.path.join(TEMP_DIR, filename)
                return __send_file(file_path)
            
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                })

        @app.route('/api/swap/image', methods=['POST'])
        def swap_image():
            try:
                # Get files from request
                target_file = request.files['target']
                source_file = request.files['source']
                
                # Save uploaded files temporarily with task ID prefix
                target_path = os.path.join(TEMP_DIR, f"{target_file.filename}")
                source_path = os.path.join(TEMP_DIR, f"{source_file.filename}")
                
                target_file.save(target_path)
                source_file.save(source_path)
                
                # Get parameters from form data
                data = {
                    'target_image': target_path,
                    'source_image': source_path,
                    'target_index': int(request.form.get('target_face_index', 0)),
                    'source_index': int(request.form.get('source_face_index', 0)),
                    'swap_all': request.form.get('swap_all', 'true').lower() == 'true',
                    'face_restore': request.form.get('face_restore', 'false').lower() == 'true',
                    'face_restore_model': request.form.get('face_restore_model', 'GFPGAN 1.3')
                }
                
                # Process image
                result_path = client.swap_image(**data)
                
                # Move result to our temp directory
                new_path = os.path.join(TEMP_DIR, os.path.basename(result_path))
                shutil.move(result_path, new_path)
                
                # Clean up input files
                os.remove(target_path)
                os.remove(source_path)
                
                return jsonify({
                    'success': True,
                    'results': [{
                        'filename': os.path.basename(new_path)
                    }]
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                })
                
        @app.route('/api/swap/video', methods=['POST'])
        def swap_video():
            try:
                # Get files from request
                target_file = request.files['target']
                source_file = request.files['source']
                
                # Save uploaded files temporarily
                target_path = os.path.join(TEMP_DIR, f"{target_file.filename}")
                source_path = os.path.join(TEMP_DIR, f"{source_file.filename}")
                
                target_file.save(target_path)
                source_file.save(source_path)
                
                # Get parameters from form data
                data = {
                    'source_image': source_path,
                    'target_video': target_path,
                    'source_index': int(request.form.get('source_face_index', 0)),
                    'face_restore': request.form.get('face_restore', 'false').lower() == 'true',
                    'face_restore_model': request.form.get('face_restore_model', 'GFPGAN 1.3')
                }
                
                # Process video
                result_path = client.swap_video(**data)
                
                # Move result to our temp directory
                new_path = os.path.join(TEMP_DIR, os.path.basename(result_path))
                shutil.move(result_path, new_path)
                
                # Clean up input files
                os.remove(target_path)
                os.remove(source_path)
                
                return jsonify({
                    'success': True,
                    'results': [{
                        'filename': os.path.basename(new_path)
                    }]
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                })
                
        @app.route('/api/detect', methods=['POST'])
        def detect_faces():
            try:
                # Get image file
                image_file = request.files['image']
                image_path = os.path.join(TEMP_DIR, f"{image_file.filename}")
                image_file.save(image_path)
                
                # Detect faces
                faces_list, total_faces, _ = client.detect_faces(image_path)
                
                # Move detected faces to our temp directory
                new_faces_list = []
                for face_path in faces_list:
                    new_path = os.path.join(TEMP_DIR, os.path.basename(face_path))
                    shutil.move(face_path, new_path)
                    new_faces_list.append(new_path)
                
                # Clean up input file
                os.remove(image_path)

                result = {
                    'total_faces': total_faces,
                    'faces': [{
                        'filename': os.path.basename(face_path)
                    } for face_path in new_faces_list]
                }
                
                return jsonify({
                    'success': True,
                    'results': result
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                })
        
        client.logger.info(f"Starting API server on {host}:{port}")
        app.run(host=host, port=port, debug=debug)
    
    except Exception as e:
        client.logger.error(f"{str(e)}")
        raise