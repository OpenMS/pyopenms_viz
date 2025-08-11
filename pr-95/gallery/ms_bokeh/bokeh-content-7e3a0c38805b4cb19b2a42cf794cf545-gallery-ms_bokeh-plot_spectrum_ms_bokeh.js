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
    
    
    const element = document.getElementById("e322a5f5-bb1f-4718-9e4e-dce5a46904ef");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid 'e322a5f5-bb1f-4718-9e4e-dce5a46904ef' but no matching script tag was found.")
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
    
      const js_urls = ["https://cdn.bokeh.org/bokeh/release/bokeh-3.7.3.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-gl-3.7.3.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-3.7.3.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-3.7.3.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-mathjax-3.7.3.min.js"];
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
                  const docs_json = '{"beb00b31-1b84-448c-ae22-43199c2c266b":{"version":"3.7.3","title":"Bokeh Application","roots":[{"type":"object","name":"Figure","id":"p3369","attributes":{"width":500,"height":500,"x_range":{"type":"object","name":"Range1d","id":"p3490","attributes":{"start":40.791199999999996,"end":241.10784}},"y_range":{"type":"object","name":"Range1d","id":"p3491","attributes":{"end":28.75}},"x_scale":{"type":"object","name":"LinearScale","id":"p3379"},"y_scale":{"type":"object","name":"LinearScale","id":"p3380"},"title":{"type":"object","name":"Title","id":"p3372","attributes":{"text":"Mass Spectrum","text_font_size":"18pt"}},"renderers":[{"type":"object","name":"GlyphRenderer","id":"p3413","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"p3404","attributes":{"selected":{"type":"object","name":"Selection","id":"p3405","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p3406"},"data":{"type":"map","entries":[["index",{"type":"ndarray","array":{"type":"bytes","data":"AAAAAAAAAAAAAAAA"},"shape":[3],"dtype":"int32","order":"little"}],["mz",{"type":"ndarray","array":{"type":"bytes","data":"O99PjZd+SUA730+Nl35JQDvfT42XfklA"},"shape":[3],"dtype":"float64","order":"little"}],["intensity",{"type":"ndarray","array":{"type":"bytes","data":"AAAAAAoAAAAAAAAA"},"shape":[3],"dtype":"int32","order":"little"}],["ion_annotation",{"type":"ndarray","array":["a+","a+","a+"],"shape":[3],"dtype":"object","order":"little"}]]}}},"view":{"type":"object","name":"CDSView","id":"p3414","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p3415"}}},"glyph":{"type":"object","name":"Line","id":"p3410","attributes":{"x":{"type":"field","field":"mz"},"y":{"type":"field","field":"intensity"},"line_color":"#7B2C65"}},"nonselection_glyph":{"type":"object","name":"Line","id":"p3411","attributes":{"x":{"type":"field","field":"mz"},"y":{"type":"field","field":"intensity"},"line_color":"#7B2C65","line_alpha":0.1}},"muted_glyph":{"type":"object","name":"Line","id":"p3412","attributes":{"x":{"type":"field","field":"mz"},"y":{"type":"field","field":"intensity"},"line_color":"#7B2C65","line_alpha":0.2}}}},{"type":"object","name":"GlyphRenderer","id":"p3425","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"p3416","attributes":{"selected":{"type":"object","name":"Selection","id":"p3417","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p3418"},"data":{"type":"map","entries":[["index",{"type":"ndarray","array":{"type":"bytes","data":"AQAAAAEAAAABAAAA"},"shape":[3],"dtype":"int32","order":"little"}],["mz",{"type":"ndarray","array":{"type":"bytes","data":"Y3/ZPXmIUkBjf9k9eYhSQGN/2T15iFJA"},"shape":[3],"dtype":"float64","order":"little"}],["intensity",{"type":"ndarray","array":{"type":"bytes","data":"AAAAABQAAAAAAAAA"},"shape":[3],"dtype":"int32","order":"little"}],["ion_annotation",{"type":"ndarray","array":["b3+","b3+","b3+"],"shape":[3],"dtype":"object","order":"little"}]]}}},"view":{"type":"object","name":"CDSView","id":"p3426","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p3427"}}},"glyph":{"type":"object","name":"Line","id":"p3422","attributes":{"x":{"type":"field","field":"mz"},"y":{"type":"field","field":"intensity"},"line_color":"#4575B4"}},"nonselection_glyph":{"type":"object","name":"Line","id":"p3423","attributes":{"x":{"type":"field","field":"mz"},"y":{"type":"field","field":"intensity"},"line_color":"#4575B4","line_alpha":0.1}},"muted_glyph":{"type":"object","name":"Line","id":"p3424","attributes":{"x":{"type":"field","field":"mz"},"y":{"type":"field","field":"intensity"},"line_color":"#4575B4","line_alpha":0.2}}}},{"type":"object","name":"GlyphRenderer","id":"p3437","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"p3428","attributes":{"selected":{"type":"object","name":"Selection","id":"p3429","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p3430"},"data":{"type":"map","entries":[["index",{"type":"ndarray","array":{"type":"bytes","data":"AgAAAAIAAAACAAAAAwAAAAMAAAADAAAABAAAAAQAAAAEAAAA"},"shape":[9],"dtype":"int32","order":"little"}],["mz",{"type":"ndarray","array":{"type":"bytes","data":"io7k8h8iWUCKjuTyHyJZQIqO5PIfIllAexSuR+FiWUB7FK5H4WJZQHsUrkfhYllALpCg+DGiWUAukKD4MaJZQC6QoPgxollA"},"shape":[9],"dtype":"float64","order":"little"}],["intensity",{"type":"ndarray","array":{"type":"bytes","data":"AAAAABkAAAAAAAAAAAAAAAwAAAAAAAAAAAAAAAYAAAAAAAAA"},"shape":[9],"dtype":"int32","order":"little"}],["ion_annotation",{"type":"ndarray","array":["c5+","c5+","c5+","c5+","c5+","c5+","c5+","c5+","c5+"],"shape":[9],"dtype":"object","order":"little"}]]}}},"view":{"type":"object","name":"CDSView","id":"p3438","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p3439"}}},"glyph":{"type":"object","name":"Line","id":"p3434","attributes":{"x":{"type":"field","field":"mz"},"y":{"type":"field","field":"intensity"},"line_color":"#91BFDB"}},"nonselection_glyph":{"type":"object","name":"Line","id":"p3435","attributes":{"x":{"type":"field","field":"mz"},"y":{"type":"field","field":"intensity"},"line_color":"#91BFDB","line_alpha":0.1}},"muted_glyph":{"type":"object","name":"Line","id":"p3436","attributes":{"x":{"type":"field","field":"mz"},"y":{"type":"field","field":"intensity"},"line_color":"#91BFDB","line_alpha":0.2}}}},{"type":"object","name":"GlyphRenderer","id":"p3449","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"p3440","attributes":{"selected":{"type":"object","name":"Selection","id":"p3441","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p3442"},"data":{"type":"map","entries":[["index",{"type":"ndarray","array":{"type":"bytes","data":"BQAAAAUAAAAFAAAA"},"shape":[3],"dtype":"int32","order":"little"}],["mz",{"type":"ndarray","array":{"type":"bytes","data":"ZmZmZmYGZEBmZmZmZgZkQGZmZmZmBmRA"},"shape":[3],"dtype":"float64","order":"little"}],["intensity",{"type":"ndarray","array":{"type":"bytes","data":"AAAAAA0AAAAAAAAA"},"shape":[3],"dtype":"int32","order":"little"}],["ion_annotation",{"type":"ndarray","array":["y9+","y9+","y9+"],"shape":[3],"dtype":"object","order":"little"}]]}}},"view":{"type":"object","name":"CDSView","id":"p3450","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p3451"}}},"glyph":{"type":"object","name":"Line","id":"p3446","attributes":{"x":{"type":"field","field":"mz"},"y":{"type":"field","field":"intensity"},"line_color":"#D73027"}},"nonselection_glyph":{"type":"object","name":"Line","id":"p3447","attributes":{"x":{"type":"field","field":"mz"},"y":{"type":"field","field":"intensity"},"line_color":"#D73027","line_alpha":0.1}},"muted_glyph":{"type":"object","name":"Line","id":"p3448","attributes":{"x":{"type":"field","field":"mz"},"y":{"type":"field","field":"intensity"},"line_color":"#D73027","line_alpha":0.2}}}},{"type":"object","name":"GlyphRenderer","id":"p3461","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"p3452","attributes":{"selected":{"type":"object","name":"Selection","id":"p3453","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p3454"},"data":{"type":"map","entries":[["index",{"type":"ndarray","array":{"type":"bytes","data":"BgAAAAYAAAAGAAAA"},"shape":[3],"dtype":"int32","order":"little"}],["mz",{"type":"ndarray","array":{"type":"bytes","data":"sp3vp8b/ZUCyne+nxv9lQLKd76fG/2VA"},"shape":[3],"dtype":"float64","order":"little"}],["intensity",{"type":"ndarray","array":{"type":"bytes","data":"AAAAAAgAAAAAAAAA"},"shape":[3],"dtype":"int32","order":"little"}],["ion_annotation",{"type":"ndarray","array":["z3+","z3+","z3+"],"shape":[3],"dtype":"object","order":"little"}]]}}},"view":{"type":"object","name":"CDSView","id":"p3462","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p3463"}}},"glyph":{"type":"object","name":"Line","id":"p3458","attributes":{"x":{"type":"field","field":"mz"},"y":{"type":"field","field":"intensity"},"line_color":"#FC8D59"}},"nonselection_glyph":{"type":"object","name":"Line","id":"p3459","attributes":{"x":{"type":"field","field":"mz"},"y":{"type":"field","field":"intensity"},"line_color":"#FC8D59","line_alpha":0.1}},"muted_glyph":{"type":"object","name":"Line","id":"p3460","attributes":{"x":{"type":"field","field":"mz"},"y":{"type":"field","field":"intensity"},"line_color":"#FC8D59","line_alpha":0.2}}}},{"type":"object","name":"GlyphRenderer","id":"p3473","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"p3464","attributes":{"selected":{"type":"object","name":"Selection","id":"p3465","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p3466"},"data":{"type":"map","entries":[["index",{"type":"ndarray","array":{"type":"bytes","data":"BwAAAAcAAAAHAAAACAAAAAgAAAAIAAAA"},"shape":[6],"dtype":"int32","order":"little"}],["mz",{"type":"ndarray","array":{"type":"bytes","data":"WvW52ooNaUBa9bnaig1pQFr1udqKDWlAWvW52oodaUBa9bnaih1pQFr1udqKHWlA"},"shape":[6],"dtype":"float64","order":"little"}],["intensity",{"type":"ndarray","array":{"type":"bytes","data":"AAAAABEAAAAAAAAAAAAAAAcAAAAAAAAA"},"shape":[6],"dtype":"int32","order":"little"}],["ion_annotation",{"type":"ndarray","array":["x4+","x4+","x4+","x4+","x4+","x4+"],"shape":[6],"dtype":"object","order":"little"}]]}}},"view":{"type":"object","name":"CDSView","id":"p3474","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p3475"}}},"glyph":{"type":"object","name":"Line","id":"p3470","attributes":{"x":{"type":"field","field":"mz"},"y":{"type":"field","field":"intensity"},"line_color":"#FCCF53"}},"nonselection_glyph":{"type":"object","name":"Line","id":"p3471","attributes":{"x":{"type":"field","field":"mz"},"y":{"type":"field","field":"intensity"},"line_color":"#FCCF53","line_alpha":0.1}},"muted_glyph":{"type":"object","name":"Line","id":"p3472","attributes":{"x":{"type":"field","field":"mz"},"y":{"type":"field","field":"intensity"},"line_color":"#FCCF53","line_alpha":0.2}}}}],"toolbar":{"type":"object","name":"Toolbar","id":"p3378","attributes":{"tools":[{"type":"object","name":"PanTool","id":"p3391"},{"type":"object","name":"WheelZoomTool","id":"p3392","attributes":{"renderers":"auto"}},{"type":"object","name":"BoxZoomTool","id":"p3393","attributes":{"dimensions":"both","overlay":{"type":"object","name":"BoxAnnotation","id":"p3394","attributes":{"syncable":false,"line_color":"black","line_alpha":1.0,"line_width":2,"line_dash":[4,4],"fill_color":"lightgrey","fill_alpha":0.5,"level":"overlay","visible":false,"left":{"type":"number","value":"nan"},"right":{"type":"number","value":"nan"},"top":{"type":"number","value":"nan"},"bottom":{"type":"number","value":"nan"},"left_units":"canvas","right_units":"canvas","top_units":"canvas","bottom_units":"canvas","handles":{"type":"object","name":"BoxInteractionHandles","id":"p3400","attributes":{"all":{"type":"object","name":"AreaVisuals","id":"p3399","attributes":{"fill_color":"white","hover_fill_color":"lightgray"}}}}}}}},{"type":"object","name":"SaveTool","id":"p3401"},{"type":"object","name":"ResetTool","id":"p3402"},{"type":"object","name":"HelpTool","id":"p3403"},{"type":"object","name":"HoverTool","id":"p3483","attributes":{"renderers":"auto","tooltips":[["m/z","@mz"],["intensity","@intensity"],["native id","@native_id"],["ion annotation","@ion_annotation"],["sequence","@sequence"]]}}]}},"toolbar_location":"above","left":[{"type":"object","name":"LinearAxis","id":"p3386","attributes":{"ticker":{"type":"object","name":"BasicTicker","id":"p3387","attributes":{"mantissas":[1,2,5]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"p3388"},"axis_label":"Intensity","axis_label_text_font_size":"16pt","major_label_policy":{"type":"object","name":"AllLabels","id":"p3389"},"major_label_text_font_size":"14pt"}}],"right":[{"type":"object","name":"Legend","id":"p3476","attributes":{"title":"Trace","click_policy":"mute","label_text_font_size":"10pt","items":[{"type":"object","name":"LegendItem","id":"p3477","attributes":{"label":{"type":"value","value":"a+"},"renderers":[{"id":"p3413"}]}},{"type":"object","name":"LegendItem","id":"p3478","attributes":{"label":{"type":"value","value":"b3+"},"renderers":[{"id":"p3425"}]}},{"type":"object","name":"LegendItem","id":"p3479","attributes":{"label":{"type":"value","value":"c5+"},"renderers":[{"id":"p3437"}]}},{"type":"object","name":"LegendItem","id":"p3480","attributes":{"label":{"type":"value","value":"y9+"},"renderers":[{"id":"p3449"}]}},{"type":"object","name":"LegendItem","id":"p3481","attributes":{"label":{"type":"value","value":"z3+"},"renderers":[{"id":"p3461"}]}},{"type":"object","name":"LegendItem","id":"p3482","attributes":{"label":{"type":"value","value":"x4+"},"renderers":[{"id":"p3473"}]}}]}}],"below":[{"type":"object","name":"LinearAxis","id":"p3381","attributes":{"ticker":{"type":"object","name":"BasicTicker","id":"p3382","attributes":{"mantissas":[1,2,5]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"p3383"},"axis_label":"mass-to-charge","axis_label_text_font_size":"16pt","major_label_policy":{"type":"object","name":"AllLabels","id":"p3384"},"major_label_text_font_size":"14pt"}}],"center":[{"type":"object","name":"Grid","id":"p3385","attributes":{"axis":{"id":"p3381"}}},{"type":"object","name":"Grid","id":"p3390","attributes":{"dimension":1,"axis":{"id":"p3386"}}},{"type":"object","name":"Label","id":"p3484","attributes":{"text":"100.5332\\nc5+\\nDMAGCH","text_color":"black","text_font_size":"13pt","x":100.5332,"y":25,"x_offset":1}},{"type":"object","name":"Label","id":"p3485","attributes":{"text":"74.1324\\nb3+\\nDMAGCH","text_color":"black","text_font_size":"13pt","x":74.1324,"y":20,"x_offset":1}},{"type":"object","name":"Label","id":"p3486","attributes":{"text":"200.4232\\nx4+\\nDMAGCH","text_color":"black","text_font_size":"13pt","x":200.4232,"y":17,"x_offset":1}},{"type":"object","name":"Label","id":"p3487","attributes":{"text":"160.2\\ny9+\\nDMAGCH","text_color":"black","text_font_size":"13pt","x":160.2,"y":13,"x_offset":1}},{"type":"object","name":"Label","id":"p3488","attributes":{"text":"101.545\\nc5+\\nDMAGCH","text_color":"black","text_font_size":"13pt","x":101.545,"y":12,"x_offset":1}},{"type":"object","name":"Span","id":"p3489","attributes":{"location":0,"line_color":"#EEEEEE","line_width":2}}],"min_border":0}}]}}';
                  const render_items = [{"docid":"beb00b31-1b84-448c-ae22-43199c2c266b","roots":{"p3369":"e322a5f5-bb1f-4718-9e4e-dce5a46904ef"},"root_ids":["p3369"]}];
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