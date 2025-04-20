diff --git a/combined_blog_workflow.py b/combined_blog_workflow_supabase.py
index abcdef0..1234567 100644
--- a/combined_blog_workflow.py
+++ b/combined_blog_workflow_supabase.py
@@ -1,6 +1,7 @@
 #!/usr/bin/env python3
 # -*- coding: utf-8 -*-
+
+from supabase import create_client  # # SUPABASE BUCKET: import Supabase client

 """
 Combined Blog Processing Workflow Script
@@ -41,6 +42,11 @@ import traceback
 from dotenv import load_dotenv
 load_dotenv(dotenv_path=ENV_FILE_PATH)
+
+# # SUPABASE BUCKET: initialize Supabase client
+supabase_url = os.getenv("SUPABASE_URL")
+supabase_key = os.getenv("SUPABASE_ANON_KEY")
+supabase     = create_client(supabase_url, supabase_key)
 

 # ---------------------------------------------------------------------
@@ -319,6 +326,12 @@ def run_module_1_blog_scraping(variables, blog_json_path):
         json.dump(scraped_data, f, indent=4)
 
+        # # SUPABASE BUCKET: upload raw blog JSON
+        with open(blog_json_path, "rb") as _f:
+            _data = _f.read()
+        res = supabase.storage.from_("agentic-output") \
+                    .upload(os.path.basename(blog_json_path), _data)
+        print(f"Uploaded {os.path.basename(blog_json_path)} → Supabase:", res)
 
     except Exception as e:
         print("Error processing the JSON file:", e)
@@ -554,6 +567,13 @@ def run_module_2_content_chunking(blog_json_path, output_chunk_path):
         with open(output_chunk_path, 'w', encoding='utf8') as f:
             json.dump(enriched_posts, f, indent=2)
 
+            # # SUPABASE BUCKET: upload chunked blog JSON
+            with open(output_chunk_path, "rb") as _f:
+                _data = _f.read()
+            res = supabase.storage.from_("agentic-output") \
+                        .upload(os.path.basename(output_chunk_path), _data)
+            print(f"Uploaded {os.path.basename(output_chunk_path)} → Supabase:", res)
+
         return True

     except Exception as chunk_error:
@@ -776,6 +796,14 @@ def run_module_4_embedding_generation(chunk_json_path, output_embeddings_path):
         with open(output_embeddings_path, "w", encoding="utf-8") as f:
             json.dump(records, f, indent=4)
 
+        # # SUPABASE BUCKET: upload embeddings JSON
+        with open(output_embeddings_path, "rb") as _f:
+            _data = _f.read()
+        res = supabase.storage.from_("agentic-output") \
+                    .upload(os.path.basename(output_embeddings_path), _data)
+        print(f"Uploaded {os.path.basename(output_embeddings_path)} → Supabase:", res)
+
         return True

     except Exception as upsert_error:

