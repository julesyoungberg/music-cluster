import js from '@eslint/js';
import prettier from 'eslint-config-prettier';
import svelte from 'eslint-plugin-svelte';
import globals from 'globals';
import ts from 'typescript-eslint';

export default ts.config(
  js.configs.recommended,
  ...ts.configs.recommended,
  ...svelte.configs['flat/recommended'],
  prettier,
  ...svelte.configs['flat/prettier'],
  {
    languageOptions: {
      globals: { ...globals.browser, ...globals.node }
    },
    rules: {
      // The API returns loosely-shaped JSON in a few places; `any` there is a
      // deliberate boundary, not an oversight.
      '@typescript-eslint/no-explicit-any': 'warn',
      '@typescript-eslint/no-unused-vars': [
        'error',
        { argsIgnorePattern: '^_', varsIgnorePattern: '^_' }
      ]
    }
  },
  {
    files: ['**/*.svelte'],
    languageOptions: {
      parserOptions: { parser: ts.parser }
    },
    rules: {
      // Naming a value inside `$effect` is how Svelte 5 registers it as a
      // dependency, so a bare expression statement is meaningful here.
      '@typescript-eslint/no-unused-expressions': 'off'
    }
  },
  {
    ignores: ['build/', '.svelte-kit/', 'dist/', 'node_modules/']
  }
);
