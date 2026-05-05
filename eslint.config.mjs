import js from '@eslint/js';
import globals from 'globals';

export default [
  js.configs.recommended,
  {
    languageOptions: {
      globals: {
        ...globals.node,
        ...globals.browser,
      },
    },
    rules: {
      "no-empty": ["error", { "allowEmptyCatch": false }],
      "no-unused-vars": ["error", { "argsIgnorePattern": "^_" }],
      "no-useless-catch": "error",
      "prefer-promise-reject-errors": "error",
    },
  },
];