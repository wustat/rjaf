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