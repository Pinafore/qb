require('./main.css');
require('./assets/css/bootstrap.min.css');
require('./assets/css/material-kit.css');
require.config({
    shim: {
        'bootstrap': {'deps': ['jquery']},
        'bootstrap-datepicker': {'deps': ['bootstrap']},
        'material-kit': {'deps': ['jquery']},
        'material': {'deps': ['jquery']},
        'nouislider': {'deps': ['jquery']}
    },
    paths: {
        'jquery': './assets/js/jquery.min.js',
        'bootstrap': './assets/js/bootstrap.min.js',
        'bootstrap-datepicker': './assets/js/bootstrap-datepicker.js',
        'material-kit': './assets/js/material-kit.js',
        'material': './assets/js/material.min.js',
        'nouislider': './assets/js/nouislider.min.js'
    }
});
var logoPath = require('./logo.svg');
var Elm = require('./Main.elm');

var root = document.getElementById('root');

Elm.Main.embed(root, logoPath);
