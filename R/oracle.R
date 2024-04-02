#' Generate an oracle solution that selects the treatment with the highest response value for each individual in validation data
#'
#' 
#' @param data validation data
#' @param y a character string indicating the column name of outcomes.
#' @param id a vector of character strings indicating the column names of IDs
#' @param trt a vector of character strings indicating the column names of treatments 
#' @return original data alongside additional columns representing the oracle-selected treatment and response variable values. These additional columns are derived from the original data based on the treatment variable's highest associated response value for each individual. 
#' @export 
#'
#' @examples 
#' data(Example_data)
#' y <- "Y"; trt <- "trt"; id <- "id"
#' 
#' Example_trainest <- Example_data %>% slice_sample(n = floor(0.3 * nrow(Example_data)))
#' Example_valid <- Example_data %>% filter(!id %in% Example_trainest$id)
#'
#' Example_oracle <- oracle(Example_valid, y, id, trt)
#' 

oracle <- function(data, y, id, trt) {
  trts <- data %>% pull(!!sym(trt)) %>% unique
  data %>%
    dplyr::select(all_of(c(id, paste0(y, trts)))) %>%
    pivot_longer(cols=paste0(y, trts), names_to=trt, names_prefix=y,
                 values_to=y) %>% group_by(id) %>%
    dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                     !!y:=max(!!sym(y)),
                     .groups="drop") %>%
    rename_with(~str_c(.,".oracle"), all_of(c(y, trt))) %>%
    inner_join(data %>% select(all_of(c(id, y))), by=id)
}