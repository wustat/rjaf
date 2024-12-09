test_that("rjaf function computes correctly", {
  data(Example_data)
  Example_trainest <- Example_data %>% slice_sample(n = floor(0.5 * nrow(Example_data)))
  Example_valid <- Example_data %>% filter(!id %in% Example_trainest$id)
  id <- "id"; y <- "Y"; trt <- "trt";  
  vars <- paste0("X", 1:3); 
  forest.reg <- rjaf(Example_trainest, Example_valid, y, id, trt, vars, 
                      clus.tree.growing = FALSE, setseed = TRUE)
  # Check if "forest.reg" has the expected class, for example, a tibble
  expect_true(is.data.frame(forest.reg), "forest.reg should be a daraframe")
})