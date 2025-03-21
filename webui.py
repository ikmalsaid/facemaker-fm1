def FacemakerWebUI(client, host: str = None, port: int = None, browser: bool = True, upload_size: str = "4MB",
                   public: bool = False, limit: int = 1, quiet: bool = False):
    """ 
    Start Facemaker WebUI with all features.
    
    Parameters:
    - client (Client): Facemaker Client instance
    - host (str): Server host
    - port (int): Server port
    - browser (bool): Launch browser automatically
    - upload_size (str): Maximum file size for uploads
    - public (bool): Enable public URL mode
    - limit (int): Maximum number of concurrent requests
    - quiet (bool): Enable quiet mode
    """
    try:
        import gradio as gr
        
        def select_faces(evt: gr.SelectData): return evt.index
        def detect_faces(image): return client.detect_faces(image) if image else None

        system_theme = gr.themes.Default(
            primary_hue=gr.themes.colors.rose,
            secondary_hue=gr.themes.colors.rose,
            neutral_hue=gr.themes.colors.zinc,
        )

        styles = '''

        ::-webkit-scrollbar {
            display: none;
        }

        ::-webkit-scrollbar-button {
            display: none;
        }

        body {
            -ms-overflow-style: none;
        }

        footer {
            display: none !important;
        }

        '''

        with gr.Blocks(analytics_enabled=False, title=f'FaceMaker FM1', css=styles, theme=system_theme).queue(default_concurrency_limit=limit) as demo:
            
            gr.Markdown(f"## <br><center>Facemaker FM1 Web UI")
            gr.Markdown(f"<center>Copyright (C) 2025 Ikmal Said. All rights reserved")
            
            with gr.Tab("Image Swap"):
                with gr.Row(equal_height=False):
                    with gr.Column(variant='panel'):
                        target_image = gr.Image(type='filepath', label='Target')
                        target_scan = gr.Button("Analyze Target Image")
                        with gr.Row():
                            target_len = gr.Number(label='Detected Faces')
                            target_sel = gr.Number(label='Selected Face')
                        target_list = gr.Gallery(type='filepath', label='Select a Face', allow_preview=False, columns=4, height=240)
                
                    with gr.Column(variant='panel'):    
                        source_image = gr.Image(type='filepath', label='Source')
                        source_scan = gr.Button("Analyze Source Image")
                        with gr.Row():
                            source_len = gr.Number(label='Detected Faces')
                            source_sel = gr.Number(label='Selected Face')
                        source_list = gr.Gallery(type='filepath', label='Select a Face', allow_preview=False, columns=4, height=240)
                    
                    with gr.Column(variant='panel'):
                        output_image = gr.Image(type='filepath', label='Output')
                        face_restore = gr.Checkbox(value=True, label='Restore face quality')
                        gfpgan_model = gr.Dropdown(choices=["GFPGAN 1.3", "GFPGAN 1.4"], value="GFPGAN 1.3", label="GFPGAN Model")
                        swap_all = gr.Checkbox(value=False, label='Swap all image faces')
                        output_swap = gr.Button("Swap from Image")

                output_swap.click(
                    client.swap_image,
                    inputs=[target_image, source_image, target_sel, source_sel, swap_all, face_restore, gfpgan_model],
                    outputs=output_image,
                    show_progress='minimal'
                )
                
                target_scan.click(
                    detect_faces,
                    inputs=[target_image],
                    outputs=[target_list, target_len, target_sel],
                    show_progress='minimal'
                    )
                
                source_scan.click(
                    detect_faces,
                    inputs=[source_image],
                    outputs=[source_list, source_len, source_sel],
                    show_progress='minimal'
                    )
                
                target_list.select(
                    select_faces,
                    inputs=None,
                    outputs=target_sel,
                    show_progress='hidden'
                    )
                
                source_list.select(
                    select_faces,
                    inputs=None,
                    outputs=source_sel,
                    show_progress='hidden'
                    )

            with gr.Tab("Video Swap"):
                with gr.Row(equal_height=False):
                    with gr.Column(variant='panel'):
                        vs_image = gr.Image(type='filepath', label='Source Image')
                        vs_scan = gr.Button("Analyze Source Image")
                        with gr.Row():
                            vs_faces = gr.Number(label='Detected Faces')
                            vs_index = gr.Number(label='Selected Face', precision=0)
                        vs_list = gr.Gallery(type='filepath', label="Select a Face", allow_preview=False, columns=4, height=240)
                    
                    with gr.Column(variant='panel'):
                        vt_file = gr.Video(label='Target Video')
                        v_enhancer = gr.Checkbox(value=True, label='Enhance face quality')
                        v_gfpgan = gr.Dropdown(choices=["GFPGAN 1.3", "GFPGAN 1.4"], value="GFPGAN 1.3", label="GFPGAN Model")
                        v_process = gr.Button("Swap from Video")

                    with gr.Column(variant='panel'):
                        v_output = gr.Video(label='Result Video')
            
                vs_scan.click(
                    detect_faces,
                    inputs=[vs_image],
                    outputs=[vs_list, vs_faces, vs_index],
                    show_progress='minimal'
                    )
                
                vs_list.select(
                    select_faces,
                    inputs=None,
                    outputs=vs_index,
                    show_progress='hidden'
                    )
                
                v_process.click(
                    client.swap_video,
                    inputs=[vs_image, vs_index, vt_file, v_enhancer, v_gfpgan],
                    outputs=v_output,
                    show_progress='minimal'
                    )
                    
        demo.launch(
            server_name=host,
            server_port=port,
            inbrowser=browser,
            max_file_size=upload_size,
            share=public,
            quiet=quiet
        )
        
    except Exception as e:
        client.logger.error(f"{str(e)}")
        raise