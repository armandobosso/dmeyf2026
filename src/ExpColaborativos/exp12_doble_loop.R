#!/usr/bin/env Rscript
# =============================================================================
# Experimento Colaborativo #12 — "Estabilidad del modelo predictivo frente al
# Undersampling"
# DMEyF 2026 · UTN · Grupo B · Bosso + Bianchini
#
# Replica el notebook 621_WorkFlow_01_junior_grupoB.ipynb en loop sobre
# (training_pct x semilla). CETERIS PARIBUS: lo unico que varia entre corridas
# es PARAM$trainingstrategy$training_pct y PARAM$semilla_primigenia.
#
# Uso:  nohup Rscript exp12_doble_loop.R > log.txt 2>&1 &
# =============================================================================

# ---------------------------------------------------------------------------
# CONFIGURACION — lo unico que cambia entre las dos maquinas
# ---------------------------------------------------------------------------

OPERADOR <- "armando"        # "armando" | "seba"

SEMILLAS <- switch(OPERADOR,
  armando = c(596027L, 830099L, 714481L, 264169L, 420479L),
  seba    = c(NA_integer_, NA_integer_, NA_integer_, NA_integer_, NA_integer_)
)

# offset del indice de semilla en el codigo de experimento:
#   armando -> semillas 01..05
#   seba    -> semillas 06..10
OFFSET_SEMILLA <- switch(OPERADOR, armando = 0L, seba = 5L)

PATH_CACHE <- "/content/buckets/b1/exp/dataset_FE_limpio.rds"
PATH_EXP   <- "/content/buckets/b1/exp"
PATH_ACUM  <- file.path(PATH_EXP, paste0("acumulador_exp12_", OPERADOR, ".csv"))

# ---------------------------------------------------------------------------
# DISENO EXPERIMENTAL — congelado, no tocar
# ---------------------------------------------------------------------------

# Orden deliberado: la comparacion primaria (1.0 vs 0.1) va primero, para que
# un corte prematuro deje pares completos. Ver nota al pie del script.
NIVELES <- list(
  list(pct = 1.00, cod = "10"),   # control
  list(pct = 0.10, cod = "01"),   # comparacion primaria pre-declarada
  list(pct = 0.40, cod = "04"),   # exploratorio
  list(pct = 0.05, cod = "05"),   # exploratorio
  list(pct = 0.01, cod = "00")    # exploratorio
)

COLS_DERIVADAS <- c("azar", "fold_train", "fold_final_train")


# ---------------------------------------------------------------------------

if (any(is.na(SEMILLAS))) {
  stop("Faltan cargar las semillas del operador '", OPERADOR, "'.")
}

require("data.table")
require("lightgbm")
require("yaml")

setDTthreads(8)

# ---------------------------------------------------------------------------
# Resume por clave: que combinaciones (pct, semilla) ya estan escritas
# ---------------------------------------------------------------------------

hechas <- character(0)
if (file.exists(PATH_ACUM)) {
  acum_previo <- fread(PATH_ACUM)
  if (nrow(acum_previo) > 0) {
    hechas <- paste(acum_previo$training_pct, acum_previo$semilla, sep = "|")
  }
  cat("Resume: ya hay", length(hechas), "corridas en el acumulador\n")
}

clave <- function(pct, semilla) paste(pct, semilla, sep = "|")

# ---------------------------------------------------------------------------
# Funcion de evaluacion del Grid Search — identica a la celda 70 del notebook
# ---------------------------------------------------------------------------

Estimar_AUC_lightgbm <- function(x) {
  param_completo <- modifyList(PARAM$lgbm$param_fijos, x)

  modelo_train <- lgb.train(
    data   = dtrain,
    valids = list(valid = dvalidate),
    eval   = "auc",
    param  = param_completo,
    verbose = -100
  )

  AUC <- modelo_train$record_evals$valid$auc$eval[[modelo_train$best_iter]]
  niter <- modelo_train$best_iter

  rm(modelo_train)
  gc(full = TRUE, verbose = FALSE)

  return(list(AUC, niter))
}

# ---------------------------------------------------------------------------
# Una corrida completa
# ---------------------------------------------------------------------------

correr_una <- function(pct, cod_nivel, semilla, idx_semilla) {

  cod_semilla  <- sprintf("%02d", idx_semilla)
  experimento  <- as.integer(paste0("9", cod_nivel, cod_semilla))
  t_inicio     <- Sys.time()

  cat("\n", strrep("=", 70), "\n", sep = "")
  cat(format(t_inicio, "%Y-%m-%d %H:%M:%S"),
      " | exp ", experimento,
      " | training_pct ", pct,
      " | semilla ", semilla, "\n", sep = "")

  # --- PARAM base -----------------------------------------------------------
  # ATENCION: los cuatro vectores de meses (validate, training, final_train,
  # future) fueron transcriptos a mano desde el notebook. VERIFICAR uno por uno
  # contra las celdas originales antes de lanzar. Un mes de mas o de menos
  # cambia todos los resultados y ninguna validacion lo detecta.
  PARAM <<- list()
  PARAM$semilla_primigenia <<- semilla
  PARAM$experimento        <<- experimento

  PARAM$trainingstrategy$validate <<- c(202105)
  PARAM$trainingstrategy$training <<- c(
    201901, 201902, 201903, 201904, 201905, 201906,
    201907, 201908, 201909, 201910, 201911, 201912,
    202001, 202002, 202003, 202004, 202005, 202006,
    202007, 202008, 202009, 202010, 202011, 202012,
    202101, 202102, 202103
  )
  PARAM$trainingstrategy$final_train <<- c(
    201901, 201902, 201903, 201904, 201905, 201906,
    201907, 201908, 201909, 201910, 201911, 201912,
    202001, 202002, 202003, 202004, 202005, 202006,
    202007, 202008, 202009, 202010, 202011, 202012,
    202101, 202102, 202103, 202104, 202105
  )
  PARAM$trainingstrategy$future    <<- c(202107)
  PARAM$trainingstrategy$positivos <<- c("BAJA+1", "BAJA+2")

  # EL parametro del experimento
  PARAM$trainingstrategy$training_pct <<- pct

  # --- carpeta de trabajo (celda 14) ---------------------------------------
  carpeta <- file.path(PATH_EXP, paste0("WF", experimento))
  dir.create(carpeta, showWarnings = FALSE)
  setwd(carpeta)

  # --- dataset limpio, desde cero, en CADA iteracion ------------------------
  t0 <- Sys.time()
  dataset <<- readRDS(PATH_CACHE)

  basura <- intersect(colnames(dataset), COLS_DERIVADAS)
  if (length(basura) > 0) {
    cat("  caché traía columnas derivadas, se eliminan:",
        paste(basura, collapse = ", "), "\n")
    dataset[, (basura) := NULL]
  }
  seg_carga <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  cat("  dataset:", nrow(dataset), "x", ncol(dataset),
      "| carga", round(seg_carga), "seg\n")

  # --- clase01 y campos_buenos (celdas 58 y 59) -----------------------------
  dataset[, clase01 := ifelse(clase_ternaria %in% PARAM$trainingstrategy$positivos, 1, 0)]

  campos_buenos <<- copy(setdiff(
    colnames(dataset), c("clase_ternaria", "clase01", "azar")
  ))

  # --- undersampling y dtrain (celda 62) ------------------------------------
  set.seed(PARAM$semilla_primigenia, kind = "L'Ecuyer-CMRG")
  dataset[, azar := runif(nrow(dataset))]

  dataset[, fold_train := foto_mes %in% PARAM$trainingstrategy$training &
      (clase_ternaria %in% c("BAJA+1", "BAJA+2") |
       azar < PARAM$trainingstrategy$training_pct)]

  n_dtrain <- dataset[fold_train == TRUE, .N]

  dtrain <<- lgb.Dataset(
    data  = data.matrix(dataset[fold_train == TRUE, campos_buenos, with = FALSE]),
    label = dataset[fold_train == TRUE, clase01],
    free_raw_data = TRUE
  )

  # --- dvalidate (celda 63) -------------------------------------------------
  dvalidate <<- lgb.Dataset(
    data  = data.matrix(dataset[foto_mes %in% PARAM$trainingstrategy$validate,
                                campos_buenos, with = FALSE]),
    label = dataset[foto_mes %in% PARAM$trainingstrategy$validate, clase01],
    free_raw_data = TRUE
  )

  cat("  dtrain:", n_dtrain, "filas\n")

  # --- param_fijos (celda 69) -----------------------------------------------
  PARAM$lgbm$param_fijos <<- list(
    objective            = "binary",
    metric               = "auc",
    first_metric_only    = TRUE,
    boost_from_average   = TRUE,
    feature_pre_filter   = FALSE,
    verbosity            = -100,
    force_row_wise       = TRUE,
    seed                 = PARAM$semilla_primigenia,
    max_bin              = 31,
    learning_rate        = 0.03,
    feature_fraction     = 0.5,
    num_iterations       = 2048,
    early_stopping_rounds = 200,
    num_leaves           = 64,
    min_data_in_leaf     = 128,
    num_threads          = 8,
    deterministic        = TRUE
  )

  # --- Grid Search (celdas 72 y 74) -----------------------------------------
  t0 <- Sys.time()
  tb_nueva <- CJ(
    num_leaves       = c(64, 128, 256, 384, 512),
    min_data_in_leaf = c(64, 256, 512, 1024, 2048),
    feature_fraction = c(0.5, 0.8)
  )

  tb_nueva[, c("AUC", "num_iterations") := Estimar_AUC_lightgbm(.SD),
           by = 1:nrow(tb_nueva)]

  seg_grid <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  fwrite(tb_nueva, file = "tb_grid_search_01.txt", sep = "\t")

  # --- ganador (celda 78) ---------------------------------------------------
  setorder(tb_nueva, -AUC)
  PARAM$out$lgbm$AUC <<- tb_nueva[1, AUC]
  PARAM$out$lgbm$mejores_hiperparametros <<- as.list(tb_nueva[1])
  PARAM$out$lgbm$mejores_hiperparametros$AUC <<- NULL

  ganador <- PARAM$out$lgbm$mejores_hiperparametros
  cat("  grid:", round(seg_grid / 60, 1), "min | ganador nl=", ganador$num_leaves,
      "mdil=", ganador$min_data_in_leaf, "ff=", ganador$feature_fraction,
      "niter=", ganador$num_iterations, "AUC=", round(PARAM$out$lgbm$AUC, 5), "\n")

  # dtrain y dvalidate viven en .GlobalEnv (se crearon con <<-). rm() sin envir
  # borraria del entorno de la funcion, donde no existen: warning y no libera.
  rm(dtrain, dvalidate, envir = .GlobalEnv)
  gc(full = TRUE, verbose = FALSE)

  # --- dfinal_train (celda 82) ----------------------------------------------
  dataset[, fold_final_train := foto_mes %in% PARAM$trainingstrategy$final_train]

  dfinal_train <- lgb.Dataset(
    data  = data.matrix(dataset[fold_final_train == TRUE, campos_buenos, with = FALSE]),
    label = dataset[fold_final_train == TRUE, clase01],
    free_raw_data = TRUE
  )

  # --- param_final (celda 84) -----------------------------------------------
  # ATENCION: se replica TAL CUAL el notebook, con c() y no modifyList().
  # Eso deja claves duplicadas (num_leaves, min_data_in_leaf, feature_fraction)
  # y LightGBM se queda con la PRIMERA, o sea con la de param_fijos.
  # Es un bug del workflow de la catedra y es parte del objeto de estudio.
  # NO CORREGIR.
  fijos <- copy(PARAM$lgbm$param_fijos)
  fijos$num_iterations        <- NULL
  fijos$early_stopping_rounds <- NULL
  param_final <- c(fijos, PARAM$out$lgbm$mejores_hiperparametros)

  # --- final model (celda 86) -----------------------------------------------
  t0 <- Sys.time()
  final_model <- lgb.train(
    data  = dfinal_train,
    param = param_final,
    verbose = -100
  )
  seg_final <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

  # hiperparametros EFECTIVOS que uso LightGBM (no los que gano el grid)
  efectivos <- final_model$params

  tb_importancia <- as.data.table(lgb.importance(final_model))
  fwrite(tb_importancia, file = "impo.txt", sep = "\t")

  # --- prediccion (celdas 92, 93, 95) ---------------------------------------
  dfuture <- dataset[foto_mes %in% PARAM$trainingstrategy$future]

  prediccion <- predict(
    final_model,
    data.matrix(dfuture[, campos_buenos, with = FALSE])
  )

  tb_prediccion <- dfuture[, list(numero_de_cliente, clase_ternaria)]
  tb_prediccion[, prob := prediccion]
  fwrite(tb_prediccion, file = "prediccion.txt", sep = "\t")

  # --- ganancia (celda 98) --------------------------------------------------
  tb_prediccion[, clase_ternaria := dfuture$clase_ternaria]
  tb_prediccion[, ganancia := -0.025]
  tb_prediccion[clase_ternaria == "BAJA+2", ganancia := 0.975]

  setorder(tb_prediccion, -prob)
  tb_prediccion[, gan_acum := cumsum(ganancia)]
  tb_prediccion[, gan_suavizada := frollmean(
    x = gan_acum, n = 400, align = "center", na.rm = TRUE, hasNA = TRUE)]

  fwrite(tb_prediccion, file = "ganancias.txt", sep = "\t")

  # --- resultado (celda 100) ------------------------------------------------
  resultado <- list()
  resultado$ganancia_suavizada_max <- max(tb_prediccion$gan_suavizada, na.rm = TRUE)
  resultado$envios <- which.max(tb_prediccion$gan_suavizada)

  PARAM$resultado <<- resultado
  write_yaml(PARAM, file = "PARAM.yml")

  seg_total <- as.numeric(difftime(Sys.time(), t_inicio, units = "secs"))

  cat("  final train:", round(seg_final / 60, 1), "min\n")
  cat("  >>> ganancia_suavizada_max =", round(resultado$ganancia_suavizada_max, 4),
      "| envios =", resultado$envios,
      "| total", round(seg_total / 60, 1), "min\n")

  # --- fila del acumulador --------------------------------------------------
  fila <- data.table(
    operador          = OPERADOR,
    experimento       = experimento,
    training_pct      = pct,
    semilla           = semilla,
    idx_semilla       = idx_semilla,
    nrow_dtrain       = n_dtrain,
    # lo que el grid creyo elegir
    grid_num_leaves       = ganador$num_leaves,
    grid_min_data_in_leaf = ganador$min_data_in_leaf,
    grid_feature_fraction = ganador$feature_fraction,
    grid_num_iterations   = ganador$num_iterations,
    grid_AUC              = PARAM$out$lgbm$AUC,
    # lo que LightGBM efectivamente uso en el Final Train
    efec_num_leaves       = efectivos$num_leaves,
    efec_min_data_in_leaf = efectivos$min_data_in_leaf,
    efec_feature_fraction = efectivos$feature_fraction,
    efec_num_iterations   = efectivos$num_iterations,
    # resultado
    ganancia_suavizada_max = resultado$ganancia_suavizada_max,
    envios                 = resultado$envios,
    # tiempos
    seg_carga  = round(seg_carga),
    seg_grid   = round(seg_grid),
    seg_final  = round(seg_final),
    seg_total  = round(seg_total),
    timestamp  = format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  )

  fwrite(fila, file = PATH_ACUM, append = file.exists(PATH_ACUM))

  rm(dfinal_train, final_model, dfuture, tb_prediccion, tb_nueva)
  rm(dataset, envir = .GlobalEnv)          # dataset tambien es global
  gc(full = TRUE, verbose = FALSE)

  invisible(NULL)
}

# ---------------------------------------------------------------------------
# DOBLE LOOP — semilla externa, nivel interno
# ---------------------------------------------------------------------------

cat(strrep("#", 70), "\n")
cat("Experimento #12 — operador:", OPERADOR, "\n")
cat("Semillas:", paste(SEMILLAS, collapse = ", "), "\n")
cat("Niveles :", paste(sapply(NIVELES, function(n) n$pct), collapse = ", "), "\n")
cat("Total   :", length(SEMILLAS) * length(NIVELES), "corridas\n")
cat("Arranca :", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat(strrep("#", 70), "\n")

t_arranque <- Sys.time()
n_corridas <- 0L

for (i in seq_along(SEMILLAS)) {

  semilla     <- SEMILLAS[i]
  idx_semilla <- i + OFFSET_SEMILLA

  for (nivel in NIVELES) {

    if (clave(nivel$pct, semilla) %in% hechas) {
      cat("skip  | pct", nivel$pct, "| semilla", semilla, "| ya estaba\n")
      next
    }

    tryCatch(
      correr_una(nivel$pct, nivel$cod, semilla, idx_semilla),
      error = function(e) {
        cat("ERROR | pct", nivel$pct, "| semilla", semilla, "|",
            conditionMessage(e), "\n")
        setwd(PATH_EXP)
        NULL
      }
    )

    n_corridas <- n_corridas + 1L
  }
}

cat("\n", strrep("#", 70), "\n", sep = "")
cat("Fin:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("Corridas en esta sesion:", n_corridas, "\n")
cat("Tiempo total:",
    round(as.numeric(difftime(Sys.time(), t_arranque, units = "hours")), 2), "hs\n")
cat("Acumulador:", PATH_ACUM, "\n")
cat(strrep("#", 70), "\n")
