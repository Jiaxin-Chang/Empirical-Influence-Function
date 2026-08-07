/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_ANNOTATION_TRAIN_DATA?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
