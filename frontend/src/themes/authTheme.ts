import { createTheme } from '@mui/material'

export const customFontDoctorProfileTheme = createTheme({
  typography: {
    fontSize: 12
  },
  components: {
    MuiInputBase: {
      styleOverrides: {
        root: {
          '&.MuiInputBase-root': {
            '&.Mui-focused fieldset': {
              borderColor: 'black'
            }
          }
        }
      }
    },
    MuiButton: {
      styleOverrides: {
        root: {
          background: 'var(--primary-color)',
          '&:hover': {
            background: 'var(--secondary-color)'
          }
        }
      }
    },
    MuiFormHelperText: {
      styleOverrides: {
        root: {
          marginLeft: '0',
          fontSize: 13
        }
      }
    },
    MuiFormLabel: {
      styleOverrides: {
        root: {
          fontSize: 16,
          color: 'var(--color-text)'
        }
      }
    },
    MuiSvgIcon: {
      variants: [
        {
          props: { fontSize: 'small' },
          style: {
            fontSize: '10px'
          }
        }
      ]
    },
    MuiRadio: {
      styleOverrides: {
        root: {
          color: 'var(--secondary-color);',
          '&.Mui-checked': {
            color: ' var(--secondary-color);'
          }
        }
      }
    },
    MuiSelect: {
      styleOverrides: {
        select: {
          width: '140px'
        }
      }
    }
  }
})

export const customFontLoginTheme = createTheme({
  typography: {
    fontSize: 12
  },
  components: {
    MuiInputBase: {
      styleOverrides: {
        root: {
          '&.MuiInputBase-root': {
            borderRadius: '30px',
            '&.Mui-focused fieldset': {
              borderColor: 'black'
            }
          }
        }
      }
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: '30px',
          background: 'var(--primary-color)',
          '&:hover': {
            background: 'var(--secondary-color)'
          }
        }
      }
    },
    MuiFormHelperText: {
      styleOverrides: {
        root: {
          marginLeft: '0',
          fontSize: 13
        }
      }
    },
    MuiFormLabel: {
      styleOverrides: {
        root: {
          fontSize: 16,
          color: 'var(--color-text)'
        }
      }
    },
    MuiSvgIcon: {
      variants: [
        {
          props: { fontSize: 'small' },
          style: {
            fontSize: '10px'
          }
        }
      ]
    },
    MuiRadio: {
      styleOverrides: {
        root: {
          color: 'var(--secondary-color);',
          '&.Mui-checked': {
            color: ' var(--secondary-color);'
          }
        }
      }
    }
  }
})
