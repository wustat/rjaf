test_that("residualize function computes correctly", {
  data(Example_data)
  
  Example_trainest <- Example_data %>% slice_sample(n = floor(0.5 * nrow(Example_data)))
  y <- "Y"
  vars <- paste0("X", 1:3)
  Example_resid <- residualize(Example_trainest, y, vars, nfold = 5, fun.rf = "ranger")
  # Check if "residualize" has the expected class, for example, a tibble
  expect_true(is.data.frame(Example_resid), "Example_resid should be a daraframe")
})