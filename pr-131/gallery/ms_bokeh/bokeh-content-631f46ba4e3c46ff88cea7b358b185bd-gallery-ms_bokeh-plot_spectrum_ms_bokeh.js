(function() {
  const fn = function() {
    'use strict';
    (function(root) {
      function now() {
        return new Date();
      }
    
      const force = false;
    
      if (typeof root._bokeh_onload_callbacks === "undefined" || force === true) {
        root._bokeh_onload_callbacks = [];
        root._bokeh_is_loading = undefined;
      }
    
    
    const element = document.getElementById("d5ffa85f-79d1-4f67-916e-7e119ecfd77d");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid 'd5ffa85f-79d1-4f67-916e-7e119ecfd77d' but no matching script tag was found.")
        }
      function run_callbacks() {
        try {
          root._bokeh_onload_callbacks.forEach(function(callback) {
            if (callback != null)
              callback();
          });
        } finally {
          delete root._bokeh_onload_callbacks
        }
        console.debug("Bokeh: all callbacks have finished");
      }
    
      function load_libs(css_urls, js_urls, callback) {
        if (css_urls == null) css_urls = [];
        if (js_urls == null) js_urls = [];
    
        root._bokeh_onload_callbacks.push(callback);
        if (root._bokeh_is_loading > 0) {
          console.debug("Bokeh: BokehJS is being loaded, scheduling callback at", now());
          return null;
        }
        if (js_urls == null || js_urls.length === 0) {
          run_callbacks();
          return null;
        }
        console.debug("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
        root._bokeh_is_loading = css_urls.length + js_urls.length;
    
        function on_load() {
          root._bokeh_is_loading--;
          if (root._bokeh_is_loading === 0) {
            console.debug("Bokeh: all BokehJS libraries/stylesheets loaded");
            run_callbacks()
          }
        }
    
        function on_error(url) {
          console.error("failed to load " + url);
        }
    
        for (let i = 0; i < css_urls.length; i++) {
          const url = css_urls[i];
          const element = document.createElement("link");
          element.onload = on_load;
          element.onerror = on_error.bind(null, url);
          element.rel = "stylesheet";
          element.type = "text/css";
          element.href = url;
          console.debug("Bokeh: injecting link tag for BokehJS stylesheet: ", url);
          document.body.appendChild(element);
        }
    
        for (let i = 0; i < js_urls.length; i++) {
          const url = js_urls[i];
          const element = document.createElement('script');
          element.onload = on_load;
          element.onerror = on_error.bind(null, url);
          element.async = false;
          element.src = url;
          console.debug("Bokeh: injecting script tag for BokehJS library: ", url);
          document.head.appendChild(element);
        }
      };
    
      function inject_raw_css(css) {
        const element = document.createElement("style");
        element.appendChild(document.createTextNode(css));
        document.body.appendChild(element);
      }
    
      const js_urls = ["https://cdn.bokeh.org/bokeh/release/bokeh-3.8.2.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-gl-3.8.2.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-3.8.2.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-3.8.2.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-mathjax-3.8.2.min.js"];
      const css_urls = [];
    
      const inline_js = [    function(Bokeh) {
          Bokeh.set_log_level("info");
        },
        function(Bokeh) {
          (function() {
            const fn = function() {
              Bokeh.safely(function() {
                (function(root) {
                  function embed_document(root) {
                  const docs_json = '{"aba67ba6-1e57-4bfe-8309-94851c525eb9":{"version":"3.8.2","title":"Bokeh Application","config":{"type":"object","name":"DocumentConfig","id":"p3398","attributes":{"notifications":{"type":"object","name":"Notifications","id":"p3399"}}},"roots":[{"type":"object","name":"Figure","id":"p3400","attributes":{"width":500,"height":500,"x_range":{"type":"object","name":"Range1d","id":"p3521","attributes":{"start":40.791199999999996,"end":241.10784}},"y_range":{"type":"object","name":"Range1d","id":"p3522","attributes":{"end":28.75}},"x_scale":{"type":"object","name":"LinearScale","id":"p3410"},"y_scale":{"type":"object","name":"LinearScale","id":"p3411"},"title":{"type":"object","name":"Title","id":"p3403","attributes":{"text":"Mass Spectrum","text_font_size":"18pt"}},"renderers":[{"type":"object","name":"GlyphRenderer","id":"p3444","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"p3435","attributes":{"selected":{"type":"object","name":"Selection","id":"p3436","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p3437"},"data":{"type":"map","entries":[["index",{"type":"ndarray","array":{"type":"bytes","data":"H4sIAAEAAAAC/2NgQAAAb8bVewwAAAA="},"shape":[3],"dtype":"int32","order":"little"}],["mz",{"type":"ndarray","array":{"type":"bytes","data":"H4sIAAEAAAAC/7O+7987vc7TwRqNBgD4LCYYGAAAAA=="},"shape":[3],"dtype":"float64","order":"little"}],["intensity",{"type":"ndarray","array":{"type":"bytes","data":"H4sIAAEAAAAC/2NgYGDgYoAAAKfaFeoMAAAA"},"shape":[3],"dtype":"int32","order":"little"}],["ion_annotation",{"type":"ndarray","array":["a+","a+","a+"],"shape":[3],"dtype":"object","order":"little"}]]}}},"view":{"type":"object","name":"CDSView","id":"p3445","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p3446"}}},"glyph":{"type":"object","name":"Line","id":"p3441","attributes":{"x":{"type":"field","field":"mz"},"y":{"type":"field","field":"intensity"},"line_color":"#7B2C65"}},"nonselection_glyph":{"type":"object","name":"Line","id":"p3442","attributes":{"x":{"type":"field","field":"mz"},"y":{"type":"field","field":"intensity"},"line_color":"#7B2C65","line_alpha":0.1}},"muted_glyph":{"type":"object","name":"Line","id":"p3443","attributes":{"x":{"type":"field","field":"mz"},"y":{"type":"field","field":"intensity"},"line_color":"#7B2C65","line_alpha":0.2}}}},{"type":"object","name":"GlyphRenderer","id":"p3456","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"p3447","attributes":{"selected":{"type":"object","name":"Selection","id":"p3448","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p3449"},"data":{"type":"map","entries":[["index",{"type":"ndarray","array":{"type":"bytes","data":"H4sIAAEAAAAC/2NkYGBghGIA++1mlAwAAAA="},"shape":[3],"dtype":"int32","order":"little"}],["mz",{"type":"ndarray","array":{"type":"bytes","data":"H4sIAAEAAAAC/0uuv2lb2RHkkIxGAwCWDCj9GAAAAA=="},"shape":[3],"dtype":"float64","order":"little"}],["intensity",{"type":"ndarray","array":{"type":"bytes","data":"H4sIAAEAAAAC/2NgYGAQYYAAAL75JIMMAAAA"},"shape":[3],"dtype":"int32","order":"little"}],["ion_annotation",{"type":"ndarray","array":["b3+","b3+","b3+"],"shape":[3],"dtype":"object","order":"little"}]]}}},"view":{"type":"object","name":"CDSView","id":"p3457","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p3458"}}},"glyph":{"type":"object","name":"Line","id":"p3453","attributes":{"x":{"type":"field","field":"mz"},"y":{"type":"field","field":"intensity"},"line_color":"#4575B4"}},"nonselection_glyph":{"type":"object","name":"Line","id":"p3454","attributes":{"x":{"type":"field","field":"mz"},"y":{"type":"field","field":"intensity"},"line_color":"#4575B4","line_alpha":0.1}},"muted_glyph":{"type":"object","name":"Line","id":"p3455","attributes":{"x":{"type":"field","field":"mz"},"y":{"type":"field","field":"intensity"},"line_color":"#4575B4","line_alpha":0.2}}}},{"type":"object","name":"GlyphRenderer","id":"p3468","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"p3459","attributes":{"selected":{"type":"object","name":"Selection","id":"p3460","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p3461"},"data":{"type":"map","entries":[["index",{"type":"ndarray","array":{"type":"bytes","data":"H4sIAAEAAAAC/2NiYGBggmJmJMyChAFTGNl2JAAAAA=="},"shape":[9],"dtype":"int32","order":"little"}],["mz",{"type":"ndarray","array":{"type":"bytes","data":"H4sIAAEAAAAC/+vqe/JJXinSoQuNrhZZ5/4wCZPWm7Dgh+EiTBoA+gkoLEgAAAA="},"shape":[9],"dtype":"float64","order":"little"}],["intensity",{"type":"ndarray","array":{"type":"bytes","data":"H4sIAAEAAAAC/2NgYGCQZEAAHiQ2G5QGAK7ZeHAkAAAA"},"shape":[9],"dtype":"int32","order":"little"}],["ion_annotation",{"type":"ndarray","array":["c5+","c5+","c5+","c5+","c5+","c5+","c5+","c5+","c5+"],"shape":[9],"dtype":"object","order":"little"}]]}}},"view":{"type":"object","name":"CDSView","id":"p3469","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p3470"}}},"glyph":{"type":"object","name":"Line","id":"p3465","attributes":{"x":{"type":"field","field":"mz"},"y":{"type":"field","field":"intensity"},"line_color":"#91BFDB"}},"nonselection_glyph":{"type":"object","name":"Line","id":"p3466","attributes":{"x":{"type":"field","field":"mz"},"y":{"type":"field","field":"intensity"},"line_color":"#91BFDB","line_alpha":0.1}},"muted_glyph":{"type":"object","name":"Line","id":"p3467","attributes":{"x":{"type":"field","field":"mz"},"y":{"type":"field","field":"intensity"},"line_color":"#91BFDB","line_alpha":0.2}}}},{"type":"object","name":"GlyphRenderer","id":"p3480","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"p3471","attributes":{"selected":{"type":"object","name":"Selection","id":"p3472","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p3473"},"data":{"type":"map","entries":[["index",{"type":"ndarray","array":{"type":"bytes","data":"H4sIAAEAAAAC/2NlYGBghWIAKU9InAwAAAA="},"shape":[3],"dtype":"int32","order":"little"}],["mz",{"type":"ndarray","array":{"type":"bytes","data":"H4sIAAEAAAAC/0tLAwK2FIc0NBoAoPQQoRgAAAA="},"shape":[3],"dtype":"float64","order":"little"}],["intensity",{"type":"ndarray","array":{"type":"bytes","data":"H4sIAAEAAAAC/2NgYGDgZYAAAL7T0OAMAAAA"},"shape":[3],"dtype":"int32","order":"little"}],["ion_annotation",{"type":"ndarray","array":["y9+","y9+","y9+"],"shape":[3],"dtype":"object","order":"little"}]]}}},"view":{"type":"object","name":"CDSView","id":"p3481","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p3482"}}},"glyph":{"type":"object","name":"Line","id":"p3477","attributes":{"x":{"type":"field","field":"mz"},"y":{"type":"field","field":"intensity"},"line_color":"#D73027"}},"nonselection_glyph":{"type":"object","name":"Line","id":"p3478","attributes":{"x":{"type":"field","field":"mz"},"y":{"type":"field","field":"intensity"},"line_color":"#D73027","line_alpha":0.1}},"muted_glyph":{"type":"object","name":"Line","id":"p3479","attributes":{"x":{"type":"field","field":"mz"},"y":{"type":"field","field":"intensity"},"line_color":"#D73027","line_alpha":0.2}}}},{"type":"object","name":"GlyphRenderer","id":"p3492","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"p3483","attributes":{"selected":{"type":"object","name":"Selection","id":"p3484","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p3485"},"data":{"type":"map","entries":[["index",{"type":"ndarray","array":{"type":"bytes","data":"H4sIAAEAAAAC/2NjYGBgg2IA1DXsdwwAAAA="},"shape":[3],"dtype":"int32","order":"little"}],["mz",{"type":"ndarray","array":{"type":"bytes","data":"H4sIAAEAAAAC/9s09/3yY/9THTah0QAzP5UEGAAAAA=="},"shape":[3],"dtype":"float64","order":"little"}],["intensity",{"type":"ndarray","array":{"type":"bytes","data":"H4sIAAEAAAAC/2NgYGDgYIAAANrdMKgMAAAA"},"shape":[3],"dtype":"int32","order":"little"}],["ion_annotation",{"type":"ndarray","array":["z3+","z3+","z3+"],"shape":[3],"dtype":"object","order":"little"}]]}}},"view":{"type":"object","name":"CDSView","id":"p3493","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p3494"}}},"glyph":{"type":"object","name":"Line","id":"p3489","attributes":{"x":{"type":"field","field":"mz"},"y":{"type":"field","field":"intensity"},"line_color":"#FC8D59"}},"nonselection_glyph":{"type":"object","name":"Line","id":"p3490","attributes":{"x":{"type":"field","field":"mz"},"y":{"type":"field","field":"intensity"},"line_color":"#FC8D59","line_alpha":0.1}},"muted_glyph":{"type":"object","name":"Line","id":"p3491","attributes":{"x":{"type":"field","field":"mz"},"y":{"type":"field","field":"intensity"},"line_color":"#FC8D59","line_alpha":0.2}}}},{"type":"object","name":"GlyphRenderer","id":"p3504","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"p3495","attributes":{"selected":{"type":"object","name":"Selection","id":"p3496","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p3497"},"data":{"type":"map","entries":[["index",{"type":"ndarray","array":{"type":"bytes","data":"H4sIAAEAAAAC/2NnYGBgh2IOJAwAFeY5nhgAAAA="},"shape":[6],"dtype":"int32","order":"little"}],["mz",{"type":"ndarray","array":{"type":"bytes","data":"H4sIAAEAAAAC/4v6uvNWF2+mQxQ2WhaTBgDbzy89MAAAAA=="},"shape":[6],"dtype":"float64","order":"little"}],["intensity",{"type":"ndarray","array":{"type":"bytes","data":"H4sIAAEAAAAC/2NgYGAQZEAAdigNAFpTpSwYAAAA"},"shape":[6],"dtype":"int32","order":"little"}],["ion_annotation",{"type":"ndarray","array":["x4+","x4+","x4+","x4+","x4+","x4+"],"shape":[6],"dtype":"object","order":"little"}]]}}},"view":{"type":"object","name":"CDSView","id":"p3505","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p3506"}}},"glyph":{"type":"object","name":"Line","id":"p3501","attributes":{"x":{"type":"field","field":"mz"},"y":{"type":"field","field":"intensity"},"line_color":"#FCCF53"}},"nonselection_glyph":{"type":"object","name":"Line","id":"p3502","attributes":{"x":{"type":"field","field":"mz"},"y":{"type":"field","field":"intensity"},"line_color":"#FCCF53","line_alpha":0.1}},"muted_glyph":{"type":"object","name":"Line","id":"p3503","attributes":{"x":{"type":"field","field":"mz"},"y":{"type":"field","field":"intensity"},"line_color":"#FCCF53","line_alpha":0.2}}}}],"toolbar":{"type":"object","name":"Toolbar","id":"p3409","attributes":{"tools":[{"type":"object","name":"PanTool","id":"p3422"},{"type":"object","name":"WheelZoomTool","id":"p3423","attributes":{"renderers":"auto"}},{"type":"object","name":"BoxZoomTool","id":"p3424","attributes":{"dimensions":"both","overlay":{"type":"object","name":"BoxAnnotation","id":"p3425","attributes":{"syncable":false,"line_color":"black","line_alpha":1.0,"line_width":2,"line_dash":[4,4],"fill_color":"lightgrey","fill_alpha":0.5,"level":"overlay","visible":false,"left":{"type":"number","value":"nan"},"right":{"type":"number","value":"nan"},"top":{"type":"number","value":"nan"},"bottom":{"type":"number","value":"nan"},"left_units":"canvas","right_units":"canvas","top_units":"canvas","bottom_units":"canvas","handles":{"type":"object","name":"BoxInteractionHandles","id":"p3431","attributes":{"all":{"type":"object","name":"AreaVisuals","id":"p3430","attributes":{"fill_color":"white","hover_fill_color":"lightgray"}}}}}}}},{"type":"object","name":"SaveTool","id":"p3432"},{"type":"object","name":"ResetTool","id":"p3433"},{"type":"object","name":"HelpTool","id":"p3434"},{"type":"object","name":"HoverTool","id":"p3514","attributes":{"renderers":"auto","tooltips":[["m/z","@mz"],["intensity","@intensity"],["native id","@native_id"],["ion annotation","@ion_annotation"],["sequence","@sequence"]],"sort_by":null}}]}},"toolbar_location":"above","left":[{"type":"object","name":"LinearAxis","id":"p3417","attributes":{"ticker":{"type":"object","name":"BasicTicker","id":"p3418","attributes":{"mantissas":[1,2,5]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"p3419"},"axis_label":"Intensity","axis_label_text_font_size":"16pt","major_label_policy":{"type":"object","name":"AllLabels","id":"p3420"},"major_label_text_font_size":"14pt"}}],"right":[{"type":"object","name":"Legend","id":"p3507","attributes":{"title":"Trace","click_policy":"mute","label_text_font_size":"10pt","items":[{"type":"object","name":"LegendItem","id":"p3508","attributes":{"label":{"type":"value","value":"a+"},"renderers":[{"id":"p3444"}]}},{"type":"object","name":"LegendItem","id":"p3509","attributes":{"label":{"type":"value","value":"b3+"},"renderers":[{"id":"p3456"}]}},{"type":"object","name":"LegendItem","id":"p3510","attributes":{"label":{"type":"value","value":"c5+"},"renderers":[{"id":"p3468"}]}},{"type":"object","name":"LegendItem","id":"p3511","attributes":{"label":{"type":"value","value":"y9+"},"renderers":[{"id":"p3480"}]}},{"type":"object","name":"LegendItem","id":"p3512","attributes":{"label":{"type":"value","value":"z3+"},"renderers":[{"id":"p3492"}]}},{"type":"object","name":"LegendItem","id":"p3513","attributes":{"label":{"type":"value","value":"x4+"},"renderers":[{"id":"p3504"}]}}]}}],"below":[{"type":"object","name":"LinearAxis","id":"p3412","attributes":{"ticker":{"type":"object","name":"BasicTicker","id":"p3413","attributes":{"mantissas":[1,2,5]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"p3414"},"axis_label":"mass-to-charge","axis_label_text_font_size":"16pt","major_label_policy":{"type":"object","name":"AllLabels","id":"p3415"},"major_label_text_font_size":"14pt"}}],"center":[{"type":"object","name":"Grid","id":"p3416","attributes":{"axis":{"id":"p3412"}}},{"type":"object","name":"Grid","id":"p3421","attributes":{"dimension":1,"axis":{"id":"p3417"}}},{"type":"object","name":"Label","id":"p3515","attributes":{"text":"100.5332\\nc5+\\nDMAGCH","text_color":"black","text_font_size":"13pt","x":100.5332,"y":25,"x_offset":1}},{"type":"object","name":"Label","id":"p3516","attributes":{"text":"74.1324\\nb3+\\nDMAGCH","text_color":"black","text_font_size":"13pt","x":74.1324,"y":20,"x_offset":1}},{"type":"object","name":"Label","id":"p3517","attributes":{"text":"200.4232\\nx4+\\nDMAGCH","text_color":"black","text_font_size":"13pt","x":200.4232,"y":17,"x_offset":1}},{"type":"object","name":"Label","id":"p3518","attributes":{"text":"160.2\\ny9+\\nDMAGCH","text_color":"black","text_font_size":"13pt","x":160.2,"y":13,"x_offset":1}},{"type":"object","name":"Label","id":"p3519","attributes":{"text":"101.545\\nc5+\\nDMAGCH","text_color":"black","text_font_size":"13pt","x":101.545,"y":12,"x_offset":1}},{"type":"object","name":"Span","id":"p3520","attributes":{"location":0,"line_color":"#EEEEEE","line_width":2}}],"min_border":0}}]}}';
                  const render_items = [{"docid":"aba67ba6-1e57-4bfe-8309-94851c525eb9","roots":{"p3400":"d5ffa85f-79d1-4f67-916e-7e119ecfd77d"},"root_ids":["p3400"]}];
                  root.Bokeh.embed.embed_items(docs_json, render_items);
                  }
                  if (root.Bokeh !== undefined) {
                    embed_document(root);
                  } else {
                    let attempts = 0;
                    const timer = setInterval(function(root) {
                      if (root.Bokeh !== undefined) {
                        clearInterval(timer);
                        embed_document(root);
                      } else {
                        attempts++;
                        if (attempts > 100) {
                          clearInterval(timer);
                          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
                        }
                      }
                    }, 10, root)
                  }
                })(window);
              });
            };
            if (document.readyState != "loading") fn();
            else document.addEventListener("DOMContentLoaded", fn);
          })();
        },
    function(Bokeh) {
        }
      ];
    
      function run_inline_js() {
        for (let i = 0; i < inline_js.length; i++) {
          inline_js[i].call(root, root.Bokeh);
        }
      }
    
      if (root._bokeh_is_loading === 0) {
        console.debug("Bokeh: BokehJS loaded, going straight to plotting");
        run_inline_js();
      } else {
        load_libs(css_urls, js_urls, function() {
          console.debug("Bokeh: BokehJS plotting callback run at", now());
          run_inline_js();
        });
      }
    }(window));
  };
  if (document.readyState != "loading") fn();
  else document.addEventListener("DOMContentLoaded", fn);
})();